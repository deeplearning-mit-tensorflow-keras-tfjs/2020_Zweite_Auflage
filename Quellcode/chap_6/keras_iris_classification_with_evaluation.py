#
# Klassifikation der Iris-Blumen mit Evaluationsmetriken mit Keras (TensorFlow 2.x)
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from pprint import pprint
import numpy as np


# Laden der iris.csv Datei
iris_data = np.loadtxt("data/iris.csv",delimiter=",",dtype="str",skiprows=1)

# Wir erstellen die Kategorien:
iris_label_array = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
label_index = 0
for label in iris_label_array :
    print(label)
    iris_data[np.where(iris_data[:,4]==label),4] = label_index
    label_index = label_index + 1

iris_data = iris_data.astype("float32")

# Wir splitten in train und evaluation Daten
# 80% train und 20% Evaluation
input_data = iris_data[:,0:4] # Spalten 0 bis 4 werden extrahier bzw. 'sepal length' 'sepal width' 'petal length' 'petal width'
output_data = iris_data[:,4].reshape(-1, 1) # Die 4. Spalte wird extrahiert und in einen Array von 1D-Array umgewandelt
output_data = to_categorical(output_data)

iris_train_input, iris_test_input, iris_train_output, iris_test_output = train_test_split(input_data, output_data, test_size=0.20)

# Aufbau des Modells mit Keras
iris_model = Sequential()
iris_model.add(Dense(5,input_shape=(4,),activation="relu"))
iris_model.add(Dense(24,activation="relu"))
iris_model.add(Dense(3,activation="softmax"))

sgd = SGD(lr=0.001)

iris_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy",tf.keras.metrics.mae])
iris_model.fit(x=iris_train_input, y=iris_train_output, batch_size=10, epochs=500, verbose=1)


# Evaluation auf Test Daten
evaluation_results = iris_model.evaluate(iris_test_input, iris_test_output)

print("Loss: {}".format(evaluation_results[0]))
print("Accuracy: {}".format(evaluation_results[1]))
print("Mean Absolute Error: {}".format(evaluation_results[2]))

# Test 
test_data = np.array([[5.1,3.5,1.4,0.2], [5.9,3.,5.1,1.8], [4.9,3.,1.4,0.2], [5.8,2.7,4.1,1.]])
predictions = iris_model.predict(test_data)
index_max_predictions = np.argmax(predictions,axis=1)

for i in index_max_predictions:
    print("Iris mit den Eigenschaften {} gehört zur Klasse: {}".format(
    test_data[i],
    iris_label_array[i]))
