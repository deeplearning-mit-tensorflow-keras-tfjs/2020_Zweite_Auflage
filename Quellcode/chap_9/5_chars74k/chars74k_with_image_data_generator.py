#
# Projekt 5 : Training eines Modells für Buchstaben- und Ziffererkennung
# mit dem ImageDataGenerator und dem 
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# Chars74K Dataset 
# (http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
# 

import os
import tensorflow as tf
import random
import numpy as np
import string
import matplotlib
import matplotlib.pyplot as plt
import gzip
import cv2


from prettytable import PrettyTable
from skimage import color, exposure, transform, io
from skimage.io import imread
from PIL import Image


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


LIST_OF_CHARS = string.digits + string.ascii_uppercase + string.ascii_lowercase

TRAIN_IMAGES = "data/GoodImg/Bmp/"  

NUM_OUTPUTS = len(LIST_OF_CHARS)
NUM_BATCHES = 32
NUM_EPOCHS = 100
IMAGE_SIZE = 28

def load_chars74k_dataset(path):
    print("Loading Dataset")
    train_images = []
    train_labels = []
 
    dir_index = 0
    for directory in sorted(os.listdir(path)):
        if(directory.startswith('.') == False):

            #print("Buchstabe: {}".format(LIST_OF_CHARS[dir_index]))
            for filename in sorted(os.listdir(path + directory)):

                if(filename.startswith('.') == False):
                    
                    pictureFilePath = path + directory + "/" + filename    
                    maskFilePath = pictureFilePath.replace('Bmp', 'Msk')
                
                    img =cv2.imread(pictureFilePath)
                    mask_img = cv2.imread(maskFilePath)

                    img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE),None,interpolation=cv2.INTER_CUBIC)
                    mask_img = cv2.resize(mask_img,(IMAGE_SIZE,IMAGE_SIZE),None,interpolation=cv2.INTER_CUBIC)
                    
                    # Mit oder ohne Maske
                    img = cv2.bitwise_and(img,mask_img)

                    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    img = img.reshape([IMAGE_SIZE, IMAGE_SIZE,1])

                    train_images.append(img)
                    train_labels.append(dir_index)

            #print(dir_index)      
            dir_index = dir_index + 1

    return np.array(train_images)/255, np.array(tf.keras.utils.to_categorical(train_labels,62))

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization( input_shape = (IMAGE_SIZE,IMAGE_SIZE,1)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape =
     (IMAGE_SIZE,IMAGE_SIZE,1)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.8))
    model.add(tf.keras.layers.Dense(NUM_OUTPUTS, activation='softmax'))
    return model

# Trainiert auf einem definierten Dataset
def train_with_generator(path):

    train_images, train_labels  = load_chars74k_dataset(TRAIN_IMAGES)
    train_images, train_labels  = shuffle(train_images,train_labels,random_state=21)
    xTrain, xTest, yTrain, yTest = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 0)

    # Imagegenerator für die Trainingsdaten
    train_imagedatagenerator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        zoom_range=[0.4,1.1],
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False)

    train_imagedatagenerator.fit(xTrain)

    # ImageDataGenerator für die Testdaten
    test_imagedatagenerator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        zoom_range=[0.5,1.0],
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False)

    test_imagedatagenerator.fit(xTest)

    optimizer = tf.keras.optimizers.Adam()

    model = build_model()
    # model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics = ["accuracy"])#

    model.fit(train_imagedatagenerator.flow(xTrain, yTrain, batch_size=NUM_BATCHES), steps_per_epoch=len(xTrain) / 32, epochs=NUM_EPOCHS,verbose=1)

    results = model.evaluate(test_imagedatagenerator.flow(xTest,yTest),verbose=1)

    print("--- Ergebnisse {} ----".format(path))
    print('Evaluation / Loss {}, Acc:{}'.format(results[0],results[1]))
    export_path = "chars74_with_image_data_generator.h5"
    model.save(export_path)
    print("💾 Modell gespeichert")

train_with_generator(TRAIN_IMAGES)