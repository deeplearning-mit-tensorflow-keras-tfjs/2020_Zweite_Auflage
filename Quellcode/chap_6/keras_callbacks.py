#
# Benutzung von Callbacks mit Keras in TensorFlow 2
#

import tensorflow as tf
import time
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from pprint import pprint

# Einfaches XOR mit Keras
input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
output_data = np.array([[0],[1],[1],[0]])

# Definition des Modells
xor_model = Sequential()
xor_model.add(Dense(1024,input_dim=2,activation="relu"))
xor_model.add(Dense(1,activation="sigmoid"))

xor_model.summary()

sgd = SGD(lr=0.01)

# Definition eines neuen Callbacks
class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print("Anfang vom Training")
        self.begin_time = time.time()
    def on_train_end(self, logs={}):
        print("Ende vom Training")
        self.training_duration = time.time() - self.begin_time
        print("Dauer des Trainings: {} sec.".format(self.training_duration))

    ''' Optional 
    def on_batch_end(self, batch, logs={}):
        print("On Batch End")
    def on_epoch_begin(self,epoch,logs={}):
        print("On Epoch Begin")
    '''

# CSV Logger Callback
csv_logger_cb = CSVLogger("xor_log.csv", separator=',', append=False)

# Eigenes Callback
my_cb = MyCallback()

# Early Stopping Callback
early_stopping_cb = EarlyStopping(monitor="loss",min_delta=0.0001,patience=20)

# Modell wird festgelegt und trainiert
xor_model.compile(loss="mean_squared_error", optimizer=sgd, metrics=[metrics.mae])
xor_model.fit(input_data, output_data, batch_size=1, epochs=10000, verbose=1,callbacks=[csv_logger_cb,my_cb,early_stopping_cb])

pprint(xor_model.predict(input_data))

