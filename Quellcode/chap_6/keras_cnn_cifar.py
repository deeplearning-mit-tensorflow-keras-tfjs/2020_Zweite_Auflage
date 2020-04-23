# -*- coding: utf-8 -*-
#
# Erstellung eines Bildklassifikators auf Basis des CIFAR-10 Dataset mit Keras (TensorFlow 2)
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np
import urllib
import matplotlib

import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# tf.enable_eager_execution()

# Variablen für das Training
BATCH_SIZE = 64
EPOCHS = 50

# Für die 10 Klassen von CIFAR-10
CIFAR_10_CLASSES = ["Flugzeug","Auto","Vogel","Katze","Hirsch","Hund", "Frosch", "Pferd","Boot","LKW"]
NUM_CLASSES = 10

# Wir laden den Datenset über Keras
(images_train, labels_train), (images_test, labels_test) = cifar10.load_data()

# Test
plt.title(CIFAR_10_CLASSES[int(labels_train[25])])
plt.imshow(images_train[25])
plt.show()

images_train = np.array(images_train,dtype="float32")
images_test = np.array(images_test,dtype="float32")

images_train /= 255 # Damit die Werte zwischen 0 und 1 bleiben
images_test /=255

labels_train = to_categorical(labels_train, NUM_CLASSES)
labels_test = to_categorical(labels_test, NUM_CLASSES)


# Beispiel von einer Subclassing eines Keras-Modells
class MyCIFARModel(tf.keras.Model):
    def __init__(self):
        super(MyCIFARModel, self).__init__()
        self.conv_1 = Conv2D(32, (3, 3), padding='same',input_shape=(32, 32, 3),activation="relu")
        self.max_pool_1 = MaxPool2D(pool_size=(2,2))
        self.conv_2 = Conv2D(64, (3, 3), padding='same')
        self.max_pool_2 = MaxPool2D(pool_size=(2,2))
        self.conv_3 = Conv2D(64, (3, 3), padding='same')
        self.max_pool_3 = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.dense_1 = Dense(512,activation="relu")
        self.softmax = Dense(10,activation='softmax')

    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.max_pool_3(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.softmax(x)

        return x   


# Variante 1: Definition des Modells mit Sequential

print("=== Variante 1 mit Sequential === ")

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32, 32, 3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dense(NUM_CLASSES,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ["accuracy"])
model.fit(images_train,labels_train, batch_size=BATCH_SIZE,epochs=EPOCHS)
scores = model.evaluate(images_test,labels_test)

print('Loss:', scores[0])
print('Accuracy:', scores[1])

tf.saved_model.save(model,"cifar_model_sequential")
model.save("cifar_model.h5")
del model

# Variante 2
print("=== Variante 2 mit Model-Subclassing === ")
model = MyCIFARModel()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ["accuracy"])
model.fit(images_train,labels_train, batch_size=BATCH_SIZE,epochs=EPOCHS)
scores = model.evaluate(images_test,labels_test)
print('Loss:', scores[0])
print('Accuracy:', scores[1])

# Wir speichern das Model mit tf.saved_model
model.save('cifar_model_subclassing', save_format='tf')

# Erzeugt aktuell eine Fehlermeldung, wenn Modell-Subclassing
# model.save("cifar_model_subclassing.h5") 
# Lösung:
model.save_weights("cifar_model_subclassing_weights")
del model

# Aus dem Google Colab von
# Francois Chollet:
# https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=gMg87Tz01cxQ
my_new_model = MyCIFARModel()
my_new_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ["accuracy"])
my_new_model.load_weights("cifar_model_subclassing_weights")

predictions = my_new_model.predict(images_train[25].reshape(-1, 32, 32, 3))
index_max_predictions = np.argmax(predictions)
print("Ergebnis: {}".format(CIFAR_10_CLASSES[index_max_predictions]))
