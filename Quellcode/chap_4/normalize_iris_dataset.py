#
# Beispiel einer Normalisierung von Daten 
# 
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = datasets.load_iris()
X = data.data
y = data.target

# Der Einfachheit halber geben wir nur die 10 ersten Eintraegen raus

X_orig = X[0:10]
print(X_orig)

# Die Werte werden auf ein Interval [0 ... 1] normalisiert ...
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_orig)

# ... und mit genau 3 Dezimalstellen ausgegeben
np.set_printoptions(precision=3,floatmode="fixed")
print(X_scaled)
