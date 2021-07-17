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
    # gray_data_ = cv2.cvtColor(data_, cv2.COLOR_BGR2GRAY)
    gray_data_ = data_[:, :, 1]
    padding_data_ = np.zeros((590, 571))
    padding_data_[3:587, 3:568] = gray_data_
    img_patch = np.empty((1, 584*565, 7, 7))
    count = 0
    for i in range(584):
        for j in range(565):
            img_patch[0, count, :, :] = padding_data_[i: i + 7, j: j + 7]
            # print(img_patch[0, count, :, :].shape, padding_data_[i * 7:i * 7 + 7, j * 7:j * 7 + 7].shape)
            count += 1
    return img_patch


def label_img_split(path):
    data_ = np.array(Image.open(path))
    img_patch = np.empty((1, 584*565))
    count = 0
    for i in range(584):
        for j in range(565):
            if data_[i, j] == 255.:
                img_patch[0, count] = 1
            else:
                img_patch[0, count] = 0
            # print(img_patch[0, count, :, :].shape, padding_data_[i * 7:i * 7 + 7, j * 7:j * 7 + 7].shape)
            count += 1
    return img_patch


class FCN:
    def __init__(self):
        self.in_shape = (7, 7, 1)
        self.lr = 0.001
        self.out_shape = 2
        self.batchsize = 64
        self.epochs = 5
        self.net = self.net_builder()

    def net_builder(self):
        input_ = keras.Input(self.in_shape, dtype='float', name='patch input')
        common = keras.layers.Conv2D(3, (3, 3), strides=(1, 1), activation='relu')(input_)
        common = keras.layers.Conv2D(7, (3, 3), strides=(1, 1), activation='relu')(common)
        common = keras.layers.Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu')(common)
        common = keras.layers.Conv2D(21, (3, 3), strides=(1, 1), padding='same', activation='relu')(common)
        common = keras.layers.Conv2D(27, (3, 3), strides=(1, 1), padding='same',  activation='relu')(common)
        common = keras.layers.BatchNormalization()(common)
        common = keras.layers.Flatten()(common)
        common = keras.layers.Dense(units=64, activation='sigmoid')(common)
        common = keras.layers.Dense(units=self.out_shape, activation='softmax')(common)

        model = keras.Model(inputs=input_, outputs=common)
        # plot_model(model, to_file='net_builder.png')
        return model

    def compulate_loss(self):
        pass

    def train_one_batch(self, train_x, train_y):
        logits = self.net.fit(train_y, train_x)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

    path_dir_img = os.path.join(path_dir, 'DRIVE', 'test', 'images').replace('\\', '/')
    file_list_img = os.listdir(path_dir_img)
    file_list_img.sort()
    img_origin = np.empty((len(file_list_img[:offset]), patch_size, 7, 7))

    path_dir_lab = os.path.join(path_dir, 'DRIVE', 'test', '1st_manual').replace('\\', '/')
    file_list_lab = os.listdir(path_dir_lab)
    file_list_lab.sort()
    img_label = np.empty((len(file_list_lab[:offset]), patch_size))

    counter = 0
    for file in file_list_img[:offset]:
        os.chdir(path_dir_img)
        print(f'image file name: {file}')
        img_iter = origin_img_split(file)
        img_origin[counter, :, :, :] = img_iter
        counter += 1
        time.time()

    counter = 0
    for file in file_list_lab[:offset]:
        os.chdir(path_dir_lab)
        print(f'label file name: {file}')
        img_iter = label_img_split(file)
        img_label[counter, :] = img_iter.reshape(-1,)
        counter += 1
        time.time()

    img_origin = (img_origin/255).squeeze().reshape((-1, 7, 7, 1))

    img_label = tf.one_hot(img_label.squeeze(), depth=2)
    db_train = tf.data.Dataset.from_tensor_slices((img_origin, img_label)).shuffle(500).batch(20)
    drive = FCN()
    drive.net.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    # Train on batch
    # for epoch in range(drive.epochs):
    #     loss_history = np.array([])
    #     acc_history = np.array([])
    #     print(f'epoch {epoch}/{drive.epochs}')
    #     for index, (train_x, train_y) in enumerate(db_train):
    #         loss, acc = drive.net.train_on_batch(train_x, train_y)
    #         loss_history = np.append(loss_history, loss)
    #         acc_history = np.append(acc_history, acc)
    #     print(f'current rpoch:{epoch}, loss: {loss_history.mean()}, accurency:{acc_history.mean()}')

    # Teain on datasets
    drive.net.fit(db_train, epochs=drive.epochs)
    pred_img = drive.net.predict(np.array(img_origin).reshape((-1, 7, 7, 1)))
    pred_data = np.array(tf.argmax(pred_img, axis=1)).reshape((584, 565))
    plt.imshow(pred_data, cmap='gray')
    plt.show()
