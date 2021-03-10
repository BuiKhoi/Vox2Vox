import os
import numpy as np
import tensorflow as tf
from models import *
from losses import *
import matplotlib.image as mpim
from sys import stdout
import glob
from sklearn.model_selection import train_test_split
from utils import DataGenerator

class TrainingOperator:
    def __init__(self, config):
        print('Start training')
        self.config = config
        self.classes = np.arange(config.n_classes)
        self.class_weights = np.load(self.config.class_weights)
        self.checkpoint_path = self.config.checkpoints_folder
        if os.path.exists(self.checkpoint_path)==False:
            os.mkdir(self.checkpoint_path)

        t1_list = sorted(glob.glob(config.data_folder + '*/*t1.nii.gz'))
        t2_list = sorted(glob.glob(config.data_folder + '*/*t2.nii.gz'))
        t1ce_list = sorted(glob.glob(config.data_folder + '*/*t1ce.nii.gz'))
        flair_list = sorted(glob.glob(config.data_folder + '*/*flair.nii.gz'))
        seg_list = sorted(glob.glob(config.data_folder + '*/*seg.nii.gz'))

        Nim = len(t1_list)
        idx = np.arange(Nim)

        idxTrain, idxValid = train_test_split(idx, test_size=config.train_test_size)
        sets = {'train': [], 'valid': []}

        for i in idxTrain:
            sets['train'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
        for i in idxValid:
            sets['valid'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
            
        self.train_gen = DataGenerator(sets['train'], augmentation=True, batch_size=self.config.batch_size)
        self.valid_gen = DataGenerator(sets['valid'], augmentation=True, batch_size=self.config.batch_size)

        self.G = Generator()
        self.D = Discriminator()
        if self.config.continue_training:
            self.G.load_weights(self.checkpoint_path + '/generator_latest.h5')
            self.D.load_weights(self.checkpoint_path + '/discriminator_latest.h5')

        if self.config.optimizer == 'adam':
            self.generator_optimizer = tf.keras.optimizers.Adam(self.config.learning_rate, beta_1=self.config.beta_1)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(self.config.learning_rate, beta_1=self.config.beta_1)
        elif self.config.optimizer == 'sgd':
            self.generator_optimizer = tf.keras.optimizers.SGD(self.config.learning_rate)
            self.discriminator_optimizer = tf.keras.optimizers.SGD(self.config.learning_rate)
        else:
            raise RuntimeError('Optimizer {} not supported'.format(self.config.optimizer))

    @tf.function
    def train_step(self, image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            gen_output = self.G(image, training=True)

            disc_real_output = self.D([image, target], training=True)
            disc_fake_output = self.D([image, gen_output], training=True)
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
            
            gen_loss, dice_loss, disc_loss_gen = generator_loss(target, gen_output, disc_fake_output, self.class_weights)

        generator_gradients = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.G.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.D.trainable_variables))
            
        return gen_loss, dice_loss, disc_loss_gen
        
    @tf.function
    def test_step(self, image, target):
        gen_output = self.G(image, training=False)

        disc_real_output = self.D([image, target], training=False)
        disc_fake_output = self.D([image, gen_output], training=False)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

        gen_loss, dice_loss, disc_loss_gen = generator_loss(target, gen_output, disc_fake_output, self.class_weights)
            
        return gen_loss, dice_loss, disc_loss_gen

    def fit(self):
        epochs = self.config.epochs
            
        Nt = len(self.train_gen)
        history = {'train': [], 'valid': []}
        prev_loss = np.inf
        
        epoch_v2v_loss = tf.keras.metrics.Mean()
        epoch_dice_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()
        epoch_v2v_loss_val = tf.keras.metrics.Mean()
        epoch_dice_loss_val = tf.keras.metrics.Mean()
        epoch_disc_loss_val = tf.keras.metrics.Mean()
        
        for e in range(epochs):
            print('Epoch {}/{}'.format(e+1,epochs))
            b = 0
            for Xb, yb in self.train_gen:
                b += 1
                losses = self.train_step(Xb, yb)
                epoch_v2v_loss.update_state(losses[0])
                epoch_dice_loss.update_state(losses[1])
                epoch_disc_loss.update_state(losses[2])
                
                stdout.write('\rBatch: {}/{} - loss: {:.4f} - dice_loss: {:.4f} - disc_loss: {:.4f}'
                            .format(b, Nt, epoch_v2v_loss.result(), epoch_dice_loss.result(), epoch_disc_loss.result()))
                stdout.flush()
            history['train'].append([epoch_v2v_loss.result(), epoch_dice_loss.result(), epoch_disc_loss.result()])
            
            for Xb, yb in self.valid_gen:
                losses_val = self.test_step(Xb, yb)
                epoch_v2v_loss_val.update_state(losses_val[0])
                epoch_dice_loss_val.update_state(losses_val[1])
                epoch_disc_loss_val.update_state(losses_val[2])
                
            stdout.write('\n               loss_val: {:.4f} - dice_loss_val: {:.4f} - disc_loss_val: {:.4f}'
                        .format(epoch_v2v_loss_val.result(), epoch_dice_loss_val.result(), epoch_disc_loss_val.result()))
            stdout.flush()
            history['valid'].append([epoch_v2v_loss_val.result(), epoch_dice_loss_val.result(), epoch_disc_loss_val.result()])
            
            # save pred image at epoch e 
            y_pred = self.G.predict(Xb)
            y_true = np.argmax(yb, axis=-1)
            y_pred = np.argmax(y_pred, axis=-1)

            canvas = np.zeros((128, 128*3))
            idx = np.random.randint(len(Xb))
            
            x = Xb[idx,:,:,64,2] 
            canvas[0:128, 0:128] = (x - np.min(x))/(np.max(x)-np.min(x)+1e-6)
            canvas[0:128, 128:2*128] = y_true[idx,:,:,64]/3
            canvas[0:128, 2*128:3*128] = y_pred[idx,:,:,64]/3
            
            fname = (self.checkpoint_path + '/pred@epoch_{:03d}.png').format(e+1)
            mpim.imsave(fname, canvas, cmap='gray')
            
            # save models
            print(' ')
            if epoch_v2v_loss_val.result() < prev_loss:
                self.G.save_weights(self.checkpoint_path + '/generator_best.h5') 
                self.D.save_weights(self.checkpoint_path + '/discriminator_best.h5')
                print("Validation loss decresaed from {:.4f} to {:.4f}. Model's weights are now saved.".format(prev_loss, epoch_v2v_loss_val.result()))
                prev_loss = epoch_v2v_loss_val.result()
            else:
                print("Validation loss did not decrese from {:.4f}.".format(prev_loss))
            print(' ')

            with open(self.config.training_log_file, 'a') as training_log_file:
                training_log_file.write('{} {} {}\n'.format(e, epoch_v2v_loss.result(), epoch_v2v_loss_val.result()))
            
            self.G.save_weights(self.checkpoint_path + '/generator_latest.h5') 
            self.D.save_weights(self.checkpoint_path + '/discriminator_latest.h5')

            # resets losses states
            epoch_v2v_loss.reset_states()
            epoch_dice_loss.reset_states()
            epoch_disc_loss.reset_states()
            epoch_v2v_loss_val.reset_states()
            epoch_dice_loss_val.reset_states()
            epoch_disc_loss_val.reset_states()
            
            del Xb, yb, canvas, y_pred, y_true, idx
            
        return history
