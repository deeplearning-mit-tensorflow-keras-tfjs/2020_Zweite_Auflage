#
# Beispiel einer Visualisierung des Olivetti-Dataset 
# 
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

olivetti = fetch_olivetti_faces(data_home="./data", shuffle=False, random_state=0, download_if_missing=True)

# Visualisierung des ersten Bilds 
plt.imshow(olivetti.data[0].reshape(64,64),cmap='gray')
plt.show()
