import cv2
import glob
import time
import numpy as np
from utils import DataGenerator, rescale_img

from models import *

class TestingOperator:

    def __init__(self, config):
        print('Start testing')
        self.config = config
        self.checkpoint_path = self.config.checkpoints_folder

        t1_list = sorted(glob.glob(config.test_folder + '*/*t1.nii.gz'))
        t2_list = sorted(glob.glob(config.test_folder + '*/*t2.nii.gz'))
        t1ce_list = sorted(glob.glob(config.test_folder + '*/*t1ce.nii.gz'))
        flair_list = sorted(glob.glob(config.test_folder + '*/*flair.nii.gz'))

        Nim = len(t1_list)
        idx = np.arange(Nim)
        test_set = []

        for i in idx:
            test_set.append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], t1_list[i]])

        self.test_gen = DataGenerator(test_set, batch_size=1)

        self.G = Generator()
        self.G.load_weights(self.checkpoint_path + '/generator_latest.h5')

    def test(self):
        data_range = self.test_gen.__len__()
        rand_idx = np.random.randint(data_range)
        Xt, yt = self.test_gen.__getitem__(rand_idx)

        gen_output = self.G(Xt, training=False).numpy()[0]
        for i in range(128):
            for j in range(4):
                cv2.imshow('Output ' + str(j), rescale_img(gen_output[:, :, i, j]))
                if cv2.waitKey(1):
                    pass
                time.sleep(self.config.show_image_delay / 1000)
        cv2.destroyAllWindows()