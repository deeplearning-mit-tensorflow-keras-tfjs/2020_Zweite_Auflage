#
# Ausgabe des Inhaltes der label-Datei (full_CNN.labels.p)
#

import pickle
import matplotlib.pyplot as plt
import numpy as np
import random as random

# Laden der Pickle Datei 
# Laden der Labels
labels = pickle.load(open("full_CNN_labels.p", "rb" ))
labels = np.array(labels)
labels = labels / 255 

#Laden der Bilder
images = pickle.load(open("full_CNN_train.p", "rb" ))
images = np.array(images)

# Die einzelnen Labels sind nur auf G (Grün) kodiert. 
# Um diese über matplot anzuzeigen, müssen noch R und B manuell hinzufügen bzw. das Array [R,G,B] gesetzt werden
def label_rgb(label_g):
    img_rgb_data = []
    width = label_g.shape[0]
    height = label_g.shape[1]
    for x in range(0,width):
        for y in range(0,height):
            green_value = label_g[x][y]
            img_rgb_data.append([0,green_value,0]) # RGB
    image_label_rgb = np.asarray(img_rgb_data).reshape(80,160,3) # wegen RGB
    return image_label_rgb

# Zeigt das Bild und das passende Label-Bild
def show_picture_and_label(index):
    plt.figure(figsize=(5,2))
    plt.subplot(1, 2, 1)
    plt.title("Bild #{}".format(str(index)))
    plt.imshow(images[index])
    plt.subplot(1, 2, 2)
    plt.title("Label #{}".format(str(index)))
    plt.imshow(label_rgb(labels[index]))
    plt.show()

show_picture_and_label(random.randint(0, len(images)))
 