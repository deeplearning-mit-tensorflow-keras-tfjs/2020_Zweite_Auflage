#
# Benutzung von Keras Callbacks zur Visualisierung des Modells und der Metriken
# Das Beispiel benutzt die Fashion-MNIST Klassifikationsaufgabe als Grundlage 
#

import tensorflow as tf
import numpy as np
import requests as requests
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, BatchNormalization, MaxPool2D, Conv2D,Flatten,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import LambdaCallback, RemoteMonitor
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# Laden der MNIST Daten
(train_data, train_labels), (eval_data, eval_labels) = mnist.load_data()
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
train_labels = to_categorical(train_labels, 10)
eval_labels = to_categorical(eval_labels,10)

# Modelaufbau 
def train_model():

    model = Sequential()
    model.add(Conv2D(32,(5,5),padding="same",name="Conv2D_1",input_shape=(28, 28,1),activation="relu"))
    model.add(MaxPool2D(padding='same',name="Max_Pooling_1",pool_size=(2,2),strides=2))

    model.add(Conv2D(64,(5,5),padding="same",name="Conv2D_2",activation="relu"))
    model.add(MaxPool2D(padding='same',name="Max_Pooling_2",pool_size=(2,2),strides=2))
    model.add(Flatten())

    model.add(Dense(1024,activation='relu',kernel_initializer='random_uniform',name="Dense_fc_1"))
    model.add(Dense(512,activation='relu',kernel_initializer='random_uniform',name="Dense_fc_2"))
    model.add(Dense(10, activation='softmax',name="Ausgabe"))

    model.compile(loss=keras.losses.categorical_crossentropy, 
    optimizer=tf.keras.optimizers.Adadelta(), metrics = ["accuracy","mse",metrics.categorical_accuracy])
    return model


model = train_model()

# Wird aufgerufen, wenn das Training beginnt
def train_begin():
    url = 'http://localhost:9000/publish/train/begin' 
    post_fields = {"model":model.to_json()}     
    request = requests.post(url, data=post_fields)


lambda_cb = LambdaCallback(on_train_begin=train_begin())
remote_cb = RemoteMonitor(root='http://localhost:9000',path="/publish/epoch/end/",send_as_json=True)

model.fit(train_data,train_labels, batch_size=64, epochs=120, verbose=1,validation_split=0.33,callbacks=[remote_cb,lambda_cb])
