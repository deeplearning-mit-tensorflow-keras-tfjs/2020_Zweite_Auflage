#
# Klassifikation der Iris-Blumen ohne Evaluationsmetriken mit Keras (TensorFlow 2.x)
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint
import numpy as np

# Laden der iris.csv Datei
iris_data = np.loadtxt("data/iris.csv",delimiter=",",dtype="str",skiprows=1)

# Wir erstellen die Kategorien:
iris_label_array = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
label_index = 0
for label in iris_label_array :
    print(label)
    # Wenn eins der Label innerhalb der tabelle gefunden wird,
    # Wir dieses mit dem label_index ersetzt
    # z.B. alle Einträge von Iris-versicolor werden mit dem Wert 1 ersetzt
    iris_data[np.where(iris_data[:,4]==label),4] = label_index
    label_index = label_index + 1

iris_data = iris_data.astype("float32")

input_data = iris_data[:,0:4] # Spalten 0 bis 4 werden extrahier bzw. 'sepal length' 'sepal width' 'petal length' 'petal width'
output_data = iris_data[:,4].reshape(-1, 1) # Die 4. Spalte wird extrahiert und in einen Array von 1D-Array umgewandelt
output_data = tf.keras.utils.to_categorical(output_data,3)

# Aufbau des Modells mit Keras
iris_model = Sequential()
iris_model.add(Dense(5,input_shape=(4,),activation="tanh"))
iris_model.add(Dense(24,activation="relu"))
iris_model.add(Dense(3,activation="softmax"))

sgd = SGD(lr=0.001)

iris_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
iris_model.fit(x=input_data, y=output_data, batch_size=10, epochs=500, verbose=1)

# Single test
test = np.array([[5.1,3.5,1.4,0.2], [5.9,3.,5.1,1.8], [4.9,3.,1.4,0.2], [5.8,2.7,4.1,1.]])
predictions = iris_model.predict(test)
index_max_predictions = np.argmax(predictions,axis=1)
print(index_max_predictions)

for i in index_max_predictions:
    print("Iris mit den Eigenschaften {} gehört zur Klasse: {}".format(
    test[i],
    iris_label_array[i]))
