#
#  Training auf Basis des KITTI-Dataset
#  Originalcode: https://github.com/6ixNugget/Multinet-Road-Segmentation
#  adaptiert für neuere Keras Versionen
#

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.misc
import re
import scipy as sp
import warnings
import time
import sys

from glob import glob
from skimage import color, exposure, transform, io
from PIL import Image


from tensorflow.keras import losses, optimizers, metrics, regularizers
from tensorflow.keras.layers import Input, Add, UpSampling2D, InputLayer, Conv2DTranspose, Dropout, BatchNormalization, MaxPooling2D, Conv2D,Flatten,Dense
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.optimizers import Adam


# Überprüfung, ob GPU vorhanden ist
# Eine GPU ist empfehlenswert, um das Modell zu trainieren
if not tf.test.gpu_device_name():
  warnings.warn('Keine GPU gefunden. Bitte eine GPU Benutzen, um das Modell zu trainieren')
else:
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Hier wollen wir nur 2 Klassen (Strasse oder nicht-Strasse) 
NUMBER_OF_CLASSES = 2

# Groesse des Eingabebildes
IMAGE_SHAPE = (160, 576) 
EPOCHS = 120
BATCH_SIZE = 1
LEARNING_RATE = 1e-5 #1e-4 
TRAINING_DATA_DIRECTORY ='./data/data_road/training'

# Extrahiert von helper.py 
# Liest die Daten vom Ordner TRAINING_DATA_DIRECTORY
def get_data(data_dir, image_shape):
    image_paths = glob(os.path.join(data_dir, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_dir, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    images = []
    gt_images = []
    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        images.append(image)
        gt_images.append(gt_image)

    return np.array(images), np.array(gt_images)


def train():

    x, y = get_data(TRAINING_DATA_DIRECTORY, IMAGE_SHAPE)
    
    inputs = Input(shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    # Block 1
    block1_conv1 = Conv2D(
        64, (3, 3), activation='relu', padding='same',
        name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    block2_conv1 = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    block3_conv1 = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv3)

    # Block 4
    block4_conv1 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)

    # Block 5
    block5_conv1 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(block5_conv2)
    block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_conv3)

    pool5_conv1x1 = Conv2D(2, (1, 1), activation='relu', padding='same')(block5_pool)
    upsample_1 = Conv2DTranspose(2, kernel_size=(4, 4), strides=(2, 2), padding="same")(pool5_conv1x1)

    pool4_conv1x1 = Conv2D(2, (1, 1), activation='relu', padding='same')(block4_pool)
    add_1 = Add()([upsample_1, pool4_conv1x1])

    upsample_2 = Conv2DTranspose(2, kernel_size=(4, 4), strides=(2, 2), padding="same")(add_1)
    pool3_conv1x1 = Conv2D(2, (1, 1), activation='relu', padding='same')(block3_pool)
    add_2 = Add()([upsample_2, pool3_conv1x1])

    upsample_3 = Conv2DTranspose(2, kernel_size=(16, 16), strides=(8, 8), padding="same")(add_2)
    output = Dense(2, activation='softmax')(upsample_3)


    # Modell wird erstellt 
    model = Model(inputs, output, name='multinet_seg')
    
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Das Modell wird gespeichert
    model.save('kitti_road_model.h5')

train()