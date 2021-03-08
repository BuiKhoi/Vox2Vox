import os
import cv2
import glob
import time
import numpy as np
from datetime import datetime
from utils import DataGenerator, rescale_img

from models import *

def uncategorical_label(label):
    mask = np.zeros((128, 128))
    for i in range(label.shape[-1] - 1):
        mask += label[:, :, i] * (i+1)
    return mask/3

class TestingOperator:

    def __init__(self, config):
        print('Start testing')
        self.config = config
        self.checkpoint_path = self.config.checkpoints_folder

        t1_list = sorted(glob.glob(config.test_folder + '*/*t1.nii.gz'))
        t2_list = sorted(glob.glob(config.test_folder + '*/*t2.nii.gz'))
        t1ce_list = sorted(glob.glob(config.test_folder + '*/*t1ce.nii.gz'))
        flair_list = sorted(glob.glob(config.test_folder + '*/*flair.nii.gz'))
        seg_list = sorted(glob.glob(config.test_folder + '*/*seg.nii.gz'))

        Nim = len(t1_list)
        idx = np.arange(Nim)
        test_set = []

        for i in idx:
            test_set.append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])

        self.test_gen = DataGenerator(test_set, batch_size=1, augmentation=True)

        self.G = Generator()
        self.G.load_weights(self.checkpoint_path + '/generator_latest.h5')

    def test(self):
        save_path = self.config.save_test_result_folder
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + '.avi'
        print(save_path)
        out_video = cv2.VideoWriter(
            save_path, 
            cv2.VideoWriter_fourcc(*'mpeg'), 
            20, (384,128)
        )
        
        data_range = self.test_gen.__len__()
        rand_idx = np.random.randint(data_range)
        Xt, yt = self.test_gen.__getitem__(rand_idx)

        gen_output = self.G(Xt, training=False).numpy()
        x = Xt[0,:,:,:,2]
        for idx in range(128):
            canvas = np.zeros((128, 128*3))
            canvas[0:128, 0:128] = (x[:, :, idx] - np.min(x[:, :, idx]))/(np.max(x[:, :, idx])-np.min(x[:, :, idx])+1e-6)
            canvas[0:128, 128:2*128] = uncategorical_label(yt[0,:,:,idx])
            canvas[0:128, 2*128:3*128] = uncategorical_label(gen_output[0,:,:,idx])
            cv2.imshow('Segmentations', canvas)
            out_video.write(canvas)
            if cv2.waitKey(1):
                pass
            time.sleep(self.config.show_image_delay / 1000)
        cv2.destroyAllWindows()
        out_video.release()