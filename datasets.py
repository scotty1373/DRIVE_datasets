# -*- coding: utf-8 -*-
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils.vis_utils import plot_model
import cv2
import matplotlib.pyplot as plt
import platform
import os

offset = 1
patch_size = 584*565

if platform.system() == 'Windows':
    path_dir = os.getcwd().replace('\\', '/')
else:
    path_dir = os.getcwd()


def origin_img_split(path):
    data_ = np.array(Image.open(path))
    gray_data_ = cv2.cvtColor(data_, cv2.COLOR_BGR2GRAY)
    padding_data_ = np.zeros((588, 567))
    padding_data_[2:586, 1:566] = gray_data_
    img_patch = np.empty((1, 584*565, 7, 7))
    count = 0
    for i in range(581):
        for j in range(560):
            img_patch[0, count, :, :] = padding_data_[i: i + 7, j: j + 7]
            # print(img_patch[0, count, :, :].shape, padding_data_[i * 7:i * 7 + 7, j * 7:j * 7 + 7].shape)
            count += 1
    return img_patch


def label_img_split(path):
    data_ = np.array(Image.open(path))
    padding_data_ = np.ones((588, 567))
    padding_data_[2:586, 1:566] = data_
    img_patch = np.empty((1, 584*565, 7, 7))
    count = 0
    for i in range(581):
        for j in range(560):
            img_patch[0, count, :, :] = padding_data_[i: i + 7, j: j + 7]
            # print(img_patch[0, count, :, :].shape, padding_data_[i * 7:i * 7 + 7, j * 7:j * 7 + 7].shape)
            count += 1
    return img_patch


class FCN:
    def __init__(self):
        self.in_shape = (7, 7, 1)
        self.lr = 0.001
        self.out_shape = 49
        self.batchsize = 64
        self.epochs = 5
        self.net = self.net_builder()

    def net_builder(self):
        input_ = keras.Input(self.in_shape, dtype='float', name='patch input')
        common = keras.layers.Conv2D(3, (3, 3), strides=(1, 1), activation='sigmoid')(input_)
        common = keras.layers.Conv2D(7, (3, 3), strides=(1, 1), activation='sigmoid')(common)
        common = keras.layers.Flatten()(common)
        common = keras.layers.Dense(units=128, activation='sigmoid')(common)
        common = keras.layers.Dense(units=self.out_shape, activation='sigmoid')(common)

        model = keras.Model(inputs=input_, outputs=common)
        # plot_model(model, to_file='net_builder.png')
        return model

    def train_iter(self, train_x, train_y):
        pass


if __name__ == '__main__':
    path_dir_img = os.path.join(path_dir, 'DRIVE', 'test', 'images').replace('\\', '/')
    file_list_img = os.listdir(path_dir_img)
    img_origin = np.empty((len(file_list_img[:offset]), patch_size, 7, 7))

    path_dir_lab = os.path.join(path_dir, 'DRIVE', 'test', '1st_manual').replace('\\', '/')
    file_list_lab = os.listdir(path_dir_lab)
    img_label = np.empty((len(file_list_lab[:offset]), patch_size, 7, 7))

    counter = 0
    for file in file_list_img[:offset]:
        os.chdir(path_dir_img)
        img_iter = origin_img_split(file)
        img_origin[counter, :, :, :] = img_iter
        counter += 1
        time.time()
    counter = 0
    for file in file_list_lab[:offset]:
        os.chdir(path_dir_lab)
        img_iter = label_img_split(file)
        img_label[counter, :, :, :] = img_iter
        counter += 1
        time.time()
    # db_train = tf.data.Dataset.from_tensor_slices((np.split(img_origin, img_origin.shape[1], axis=1),
    #                                               np.split(img_label, img_label.shape[1], axis=1)))
    drive = FCN()
    drive.net.compile(loss=keras.losses.binary_crossentropy, metrics=['accuracy', 'loss'])
    for epoch in range(drive.epochs):




    time.time()
