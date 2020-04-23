# 
# Keras Metriken mit Bokeh darstellen (TensorFlow 2)
#

import tensorflow as tf
import numpy as np
import requests as requests
import matplotlib.pyplot as plt 

from PIL import Image
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Input, InputLayer, BatchNormalization, MaxPool2D, Conv2D,Flatten,Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from bokeh.plotting import figure, output_file, show


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

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ["accuracy","mse",tf.keras.metrics.categorical_accuracy]) 
    model_train_history = model.fit(train_data,train_labels, batch_size=64, epochs=100, verbose=1,validation_split=0.33)
    model.save('my_model.h5')
    return model_train_history



model_train_history = train_model()


# Liste alle verfügbaren History
print(model_train_history.history.keys())

## Benutzung von Bokeh ##
output_file("keras_metrics.html")

p = figure(title="Keras Metriken",plot_width=1200, plot_height=400,tools="pan,wheel_zoom,hover,reset",sizing_mode="scale_both")

# Loss 
x_train_loss_axis = np.arange(0,len(model_train_history.history['loss'])) 
y_train_loss_axis = model_train_history.history['loss']

# Loss (validation)
x_validation_loss_axis = np.arange(0,len(model_train_history.history['val_loss'])) 
y_validation_loss_axis = model_train_history.history['val_loss']

# Accuracy
x_train_acc_axis = np.arange(0,len(model_train_history.history['accuracy']))
y_train_acc_axis = model_train_history.history['accuracy']

# Accuracy (validation)
x_validation_acc_axis = np.arange(0,len(model_train_history.history['val_accuracy']))
y_validation_acc_axis = model_train_history.history['val_accuracy']

# Labels für die jeweilige Axis
p.xaxis.axis_label = "Epochs"
p.yaxis.axis_label = "Wert"

# Kurve werden generiert
p.line(x_train_loss_axis,y_train_loss_axis,legend="Loss / Training",line_color="red",line_width=2,alpha=0.5)
p.line(x_validation_loss_axis,y_validation_loss_axis,legend="Loss / Validation",line_color="red",line_width=2)

p.line(x_train_acc_axis,y_train_acc_axis,legend="Accuracy / Training",line_color="green",line_width=2,alpha=0.5)
p.line(x_validation_acc_axis,y_validation_acc_axis,legend="Accuracy / Validation",line_color="green",line_width=2)

# Die HTML-Datei wird generiert
show(p)
