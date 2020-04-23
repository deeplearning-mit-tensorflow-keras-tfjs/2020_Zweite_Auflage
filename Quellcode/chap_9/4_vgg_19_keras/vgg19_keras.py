#
# Projekt 4: Benutzung von VGG-19 mit Keras (TensorFlow 2)
#

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt 

from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import load_model
from PIL import Image

# Initialisierung des VGG19 Modells
model = VGG19(include_top=True,weights='imagenet')
model.summary()

def get_imagenet_class(index):
    classes = json.load(open("imagenet_class_index.json")) 
    # imagenet_class_index.json muss von folgender URL heruntergeladen werden:
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    return classes[str(index)][1]

# Vorhersage für ein Bild
def predict_image(img):
    img = np.expand_dims(img,axis=0)
    image_net_index =  np.argmax(model.predict(img))
    return get_imagenet_class(image_net_index)

# Anzeige des Bildes mit Titel
def show_image(img, title):
    plt.title("Erkannt : {}\n".format(title))
    plt.axis('off')
    plt.imshow(img,interpolation='bicubic')
    plt.show()

#Test
img_path = "./img/cat.jpg"
jpgfile = np.array(Image.open(img_path).convert('RGB').resize((224, 224))) # * 255
prediction = predict_image(jpgfile)
show_image(jpgfile,prediction)