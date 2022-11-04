from tensorflow.keras.utils import load_img

import pandas as pd 
import numpy as np
import os

class DataLoader:
    def __init__(self):
        self.folder_path = "D:\\dataset\\Label_harry\\classification\\Circuit"
        self.target_size = 224

    def load_data(self, folder_path):
        label_list = []
        imgs = []
        for label in os.listdir(folder_path):
            for image in os.listdir(folder_path + "\\" + label):
                label_list.append(label)
                img = np.array(load_img(folder_path + "\\" + label + "\\" + image, target_size=(self.target_size, self.target_size)))
                imgs.append(img)
        return imgs, label_list