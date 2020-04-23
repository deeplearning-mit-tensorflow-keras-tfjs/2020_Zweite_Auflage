#
#  Einfaches CNN für Fashion-MNIST mit TensorFlow 2.x
#

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import gzip

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Input
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorboard.plugins.hparams import api as hp

# Fashion Klassen
fashion_class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Labels und Daten werden hier geladen
def load_fashion_data():
    
    # Trainingsdaten
    ## Labels (Ausgabe)
    with gzip.open("data/fashion/train-labels-idx1-ubyte.gz") as path:
        train_labels = np.frombuffer(path.read(),np.uint8,offset=8)

    ## Bilder (Eingabe)
    with gzip.open("data/fashion/train-images-idx3-ubyte.gz") as path:
        train_images = np.frombuffer(path.read(),np.uint8,offset=16).reshape(len(train_labels),28,28)/255.0

    # Testdaten 

    ## Labels (Ausgabe)
    with gzip.open("data/fashion/t10k-labels-idx1-ubyte.gz") as path:
        test_labels = np.frombuffer(path.read(),np.uint8,offset=8)

    ## Bilder (Eingabe)
    with gzip.open("data/fashion/t10k-images-idx3-ubyte.gz") as path:
        test_images = np.frombuffer(path.read(),np.uint8,offset=16).reshape(len(test_labels),28,28)/255.0

    test_images = np.reshape(test_images,(-1, 28, 28, 1))
    train_images = np.reshape(train_images,(-1, 28, 28, 1))


    seed = 42 
    # Shuffle der Testdaten
    train_images, train_labels = shuffle(train_images, train_labels,random_state=seed)

    # Shuffle der Testdaten
    test_images, test_labels = shuffle(test_images, test_labels,random_state=seed)

    # One Hot Encoding (10 Kategorien)
    train_labels = tf.keras.utils.to_categorical(train_labels,10) 
    test_labels = tf.keras.utils.to_categorical(test_labels,10)

    return train_images, train_labels, test_images, test_labels


# Laden der Daten
train_images, train_labels, test_images, test_labels = load_fashion_data()


def get_test_image(index):
    return np.reshape(test_images[index],(-1,28,28,1))


def show_test_image(index):
    plt.title(fashion_class_labels[np.argmax(test_labels[index])])
    plt.imshow(np.reshape(test_images[index],(28,28)),cmap='Greys')
    plt.show()

# Hyperparamaters
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_NUM_FILTER_FIRST_CONV = hp.HParam('num_filter_first_conv', hp.Discrete([16, 64]))
HP_NUM_FILTER_SECOND_CONV = hp.HParam('num_filter_second_conv', hp.Discrete([16, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

 # Ein einfaches Model 
def fashion_model(hParams):

    model = tf.keras.models.Sequential()

    # Convolutional Layer #1  
    model.add(tf.keras.layers.Conv2D(filters= hParams[HP_NUM_FILTER_FIRST_CONV],kernel_size=[5, 5], activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))

    # Convolutional Layer #2  
    model.add(tf.keras.layers.Conv2D(filters=hParams[HP_NUM_FILTER_SECOND_CONV],kernel_size=[5, 5], activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hParams[HP_NUM_UNITS]))
    model.add(tf.keras.layers.Dropout(hParams[HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

train_summary_writer = tf.summary.create_file_writer("logs")


# Trainieren vom Model mit hParams 
def train_fashion_model(hParams):
    model = fashion_model(hParams)
    model.compile(optimizer="adam",loss= tf.keras.losses.categorical_crossentropy,metrics=["accuracy"])
    model.fit(train_images,train_labels,epochs=1,batch_size=32, validation_split=0.1)
    accuracy = model.evaluate(test_images, test_labels)[1]
    return accuracy

# Wir iterieren über alle angelegtes hparameters
i=0
for num_filter_first_conv in HP_NUM_FILTER_FIRST_CONV.domain.values:
    for num_filter_second_conv in HP_NUM_FILTER_SECOND_CONV.domain.values:
        for num_units in HP_NUM_UNITS.domain.values:
            for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                for optimizer in HP_OPTIMIZER.domain.values:
                    
                    #  hParams, die in TensorBoard erscheinen sollen
                    hparams = {
                        HP_NUM_FILTER_FIRST_CONV: num_filter_first_conv,
                        HP_NUM_FILTER_SECOND_CONV: num_filter_second_conv,
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                       
                    }
                    with tf.summary.create_file_writer("logs/hparams_fashion_"+str(i)).as_default():
                        hp.hparams(hparams) 
                        accuracy = train_fashion_model(hparams)
                        tf.summary.scalar("Genauigkeit / Accuracy", accuracy, step=1)
                        i = i +1 