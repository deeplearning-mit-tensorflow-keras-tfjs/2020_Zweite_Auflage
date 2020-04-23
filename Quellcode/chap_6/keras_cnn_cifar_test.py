# -*- coding: utf-8 -*-
#
# Laden und Benutzung eines gespeicherten Modells mit Keras 
#

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import urllib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# CIFAR-10 Klassen
CIFAR_10_CLASSES = ["Flugzeug","Auto","Vogel","Katze","Hirsch","Hund", "Frosch", "Pferd","Boot","LKW"]

# Ein Bild wird von Wikimedia heruntergeladen
test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/VW_Polo_Genf_2018.jpg/1280px-VW_Polo_Genf_2018.jpg"
#test_image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8b/Hauskatze_filou.jpg"
#test_image_url = "https://upload.wikimedia.org/wikipedia/commons/f/f8/Pernod_Al_Ariba_0046b.jpg"

urllib.request.urlretrieve(test_image_url, "bild.jpg")

# Das Bild wird für das Modell auf 32x32 Pixel reduziert 
# und die Pixelwerte normalisiert
test_image = Image.open("bild.jpg")
test_image = test_image.resize((32,32), Image.ANTIALIAS)
test_image = np.array(test_image,dtype="float32")
test_image /= 255
test_image = test_image.reshape(-1, 32, 32, 3)


# Vorhersage vom Modell
model = load_model('cifar_model.h5')
# Alternativ: 
# model = load_model("cifar_model")

predictions = model.predict(test_image) # Das Ergebnis müsste ungefähr bei 10 liegen
index_max_predictions = np.argmax(predictions)

# Darstellung des Ergebnis
plt.title("Ergebnis: {}".format(CIFAR_10_CLASSES[index_max_predictions]))
plt.imshow(test_image[0].reshape(32,32,3))
plt.show()
