#
# Projekt 1: Verkehrszeichenerkennung mit Keras
#

import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf-gpu
import numpy as np
import csv
import PIL

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses,  metrics, regularizers, utils
from tensorflow.keras.layers import InputLayer, Dropout, BatchNormalization, MaxPooling2D, Conv2D,Flatten,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from pandas import read_csv
from skimage import color, exposure, transform, io
from PIL import Image
from textwrap import wrap
from sklearn.utils import shuffle

# Unter MacOS muss das Backend von matplotlib  veraendert werden, 
# ansonsten gibt es eine Fehlermeldung
# matplotlib.use('TkAgg')

IMG_SIZE = 32
MAX_PICTURES_PER_CLASS = 200
NUM_BATCHES = 64
NUM_EPOCHS = 40 

TRAIN_IMAGES_PATH = './img/train'
TEST_IMAGES_PATH = "./img/test"

 # Das erste Bild von jeder Kategorie wird zwischengespeichert
 # und angezeigt
first_images = []
num_roadsign_classes = 0  

# Funktion, die das Bild zu einer bestimmten Größe skaliert, 
# da jedes Bild innerhalb der Verzeichnisse unterschiedlichen Grössen
# besitzen

def preprocess_image(img):
    img = np.asarray(img)
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

# Funktion adaptiert von readTrafficSigns.py vom Institut für Neuroninformatik    
def load_roadsigns_data(rootpath):

    images = [] # Array von Bilder images
    labels = [] # Array von corresponding labels
    
    nbOfImages =0
    global num_roadsign_classes
    
    #  Hinweis: unter MacOS wird der versteckte Ordner
    # ".DS_STORE" mitgezählt
    num_roadsign_classes = len([pic for pic in os.listdir("./img/train") if not pic.startswith(".")])

    # Die Schleife geht über alle Verzeichnisse
    for c in range(0,num_roadsign_classes):
        prefix = rootpath + '/' + format(c, '05d') + '/' # Unterverzeichnis für die Kategorie
        gt_file = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gt_reader = csv.reader(gt_file, delimiter=';') 
        next(gt_reader) # Iterator
        
         # Jede zeile der GT-XXXXX.csv Datei wird eingelesen
        for row_index, row in enumerate(gt_reader):
            jpgfile = Image.open(prefix + row[0])
            nbOfImages +=1

            if(row_index>=MAX_PICTURES_PER_CLASS):
                break
            # Wir wollen nicht durch alle Bilder pro Verzeichnis durchgehen (das Training kann länger dauern)
            # wir setzten eine Grenze von 250 zu analysierenden Bildern pro 
            # Verzeichnis. Erhöhen Sie den Wert von MAX_PICTURE_PER_CLASS,
            # wenn Sie alle Bilder berücksichtigen möchten
            gs_image = preprocess_image(jpgfile) # Skaliert das Bild
            images.append(gs_image) 
            labels.append(row[7]) # die Spalte 8 beinhaltet das Label

            if(row_index==0):
                first_images.append(gs_image)
            
            if(nbOfImages%1000 == 0):
                print("Analysierte Bilder :{}".format(nbOfImages))

        gt_file.close()

    return np.array(images)/255.0, np.array(to_categorical(labels, num_classes=num_roadsign_classes))

# Gibt den Namen der Kategorie zurück
def get_roadsign_name(index):
    return sign_names.values[index][1]

# Ausgabe der Bilder
def display_roadsigns_classes():
    plt.rc('font', size=6)    
    plt.rcParams["figure.figsize"] = (10,10)
    fig, axarr = plt.subplots(6,8)
    num = 0
    for i in range(0,6):
        for p in range(0,8):
            axarr[i][p].axis('off')
            if(num<num_roadsign_classes):
                axarr[i,p].imshow(first_images[num],interpolation='nearest')
                roadsign_name = "\n".join(wrap(get_roadsign_name(num),15))
                axarr[i,p].set_title("[" + str(num) + "]\n"+roadsign_name)
                num +=1
    fig.suptitle('German Traffic Sign Recognition Benchmark',fontsize=16,fontweight="bold")
    plt.subplots_adjust(hspace=1)
    plt.show()

# Aufbau des Keras Modells 
def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization( input_shape = (IMG_SIZE,IMG_SIZE,3)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape =
     (IMG_SIZE,IMG_SIZE,3)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(num_roadsign_classes, activation='softmax'))
    return model

# Namen der Verkehrsschilder werden geladen
sign_names = read_csv('road_signs_names.csv', delimiter=',',dtype=None)

# Wir laden die Bilder und die Labels dank der load_roadsigns_data() Funktion
images, labels = load_roadsigns_data(TRAIN_IMAGES_PATH)
images, labels  = shuffle(images,labels,random_state=42)
trainImages, valImages, trainLabels, valLabels = train_test_split(images, labels, test_size = 0.2, random_state = 42)

# Die Verkehrsschilder werden angezeigt
display_roadsigns_classes()

optimizer = Adam(lr=0.001)
model = build_model()

# Da wir ein Klassifikationsaufgabe haben, verwenden wir categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ["accuracy"])#,"mae"])

# Training des Modells
model.fit(trainImages,trainLabels, epochs=NUM_EPOCHS, batch_size=NUM_BATCHES,verbose=1,validation_data=(valImages,valLabels),shuffle=True)

# Wir laden nun das Testdataset
test_images, test_labels = load_roadsigns_data(TEST_IMAGES_PATH)
test_images, test_labels  = shuffle(test_images,test_labels,random_state=42)

scores = model.evaluate(test_images,test_labels,verbose=1)

print('Loss:', scores[0])
print('Accuracy:', scores[1])

# Das Model wird gespeichert
model.save('road_signs_model.h5')