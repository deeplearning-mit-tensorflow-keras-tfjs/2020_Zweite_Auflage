#
# Benutzung von tf.summary.image() 
# 

import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar100
from keras.datasets import fashion_mnist 

# Laden des CIFAR-Datensatz
(cifar_images_train, labels_train), (images_test, labels_test) = cifar100.load_data(label_mode='fine')

# Laden des Fashion-Datensatz

(fashion_images_train,fashion_labels_train),(fashion_images_test,fashion_labels_test) = fashion_mnist.load_data()

fashion_images_train = fashion_images_train.reshape(-1, 28, 28, 1)

NUM_OF_IMAGES = 1000

# Ausgabe von N Bilder aus dem Datensatz
image_summary_writer =  tf.summary.create_file_writer("./logs/cifar")
fashion_summary_writer =  tf.summary.create_file_writer("./logs/fashion")

# CIFAR
with image_summary_writer.as_default():
    tf.summary.image("CIFAR - Bilder", cifar_images_train[0:NUM_OF_IMAGES],step=0,max_outputs=NUM_OF_IMAGES,description="Bilder vom CIFAR-Dataset")

with fashion_summary_writer.as_default():
    # Fashion MNIST
    tf.summary.image("Fashion MNIST - Bilder", fashion_images_train[0:NUM_OF_IMAGES],step=0,max_outputs=NUM_OF_IMAGES,description="Bilder vom Fashion MNIST-Dataset")
