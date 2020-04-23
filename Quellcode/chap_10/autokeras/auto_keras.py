#
# Beispiel für die Benutzung von AutoKeras
#

from tensorflow.keras.datasets import mnist
import autokeras as ak
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

# Instanzierung des ImageClassifiers von AutoKeras
clf = ak.ImageClassifier(max_trials=10) # hier wird 10 mal versucht

# Ähnlich wie bei Keras, wird hier eine fit() Funktion benutzt
clf.fit(x_train, y_train)

# Wenn ein Model gefunden wurde, wird es erneut trainiert
clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)

# Ähnlich Keras kann hier die predict()-Funktion aufgerufen werden
results = clf.predict(x_test)