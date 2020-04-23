#
# Projekt 4: Benutzung von Keras Applications
#
import os
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt 
import pprint

from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from PIL import Image
from sklearn.metrics import classification_report
from tabulate import tabulate

# Initialisierung des ausgwehählten Modells
def init_model(model_name):
    if(model_name == "VGG19"):
        return VGG19(include_top=True,weights='imagenet')
    if(model_name == "VGG16"):
        return tf.keras.applications.VGG16(include_top=True,weights='imagenet')
    if(model_name == "ResNet50"):
        return ResNet50(include_top=True,weights="imagenet")
    if(model_name == "DenseNet201"):
        return DenseNet201(include_top=True,weights="imagenet")
    if(model_name == "DenseNet121"):
        return DenseNet121(include_top=True,weights="imagenet")
    if(model_name == "InceptionResNetV2"):
        return InceptionResNetV2(include_top=True,weights="imagenet")


def get_imagenet_class(index):
    # imagenet_class_index.json muss von folgender URL heruntergeladen werden:
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    classes = json.load(open("imagenet_class_index.json"))
    return classes[str(index)][1]

# Vorhersage für ein Bild
def predict_image(model, img):
    img = np.expand_dims(img,axis=0)
    image_net_index =  np.argmax(model.predict(img))
    return get_imagenet_class(image_net_index)

# Top n Vorhersage
def predict_top_image(model,img,top_value=5):
    img = np.expand_dims(img,axis=0)
    predictions = model.predict(img)

    # hier werden die n ersten höchsten Indezes gesucht. 
    # Die Anzahl der zurückzugebende Ergebnisse werden durch den Parameter top_value bestimmt
    # Die Numpy-Funktion argpartition wird diese Werte innerhalb vom np.array predictions selektieren
    # Die von model.predict() zurückgelieferten Indizes sind nicht absteigend sortiert. 
   
    class_indexes = np.argpartition(predictions[0], -top_value)[-top_value:]

    pred = np.array(predictions[0][class_indexes])
    ind = pred.argsort()

    # Die Arrays class_index und pred sind nicht absteigend bzw. nach Relevanz sortiert: 
    # das wird durch die Angabe von [::-1] ( ähnlich einer Reverse-Funktion ) gelöst 
    return [class_indexes[ind][::-1], pred[ind][::-1]*100]  #class_indexes


# Anzeige des Bildes mit Titel
def show_image(img, title):
    plt.title("Erkannt : {}\n".format(title))
    plt.axis('off')
    plt.imshow(img,interpolation='none')
    plt.show()


# Testbilder
# Bitte anpassen
img_path = "./img/cat.jpg"
jpgfile = np.array(Image.open(img_path).convert('RGB').resize((224, 224))) # * 255


# Liste der Modelle, die wir benutzen werden
model_names = ["VGG16","VGG19","ResNet50"]
TOP_VALUE = 5

# Für jedes Model werden wir die Ausgabe von model.predict generieren und anzeigen lassen
for model_name in model_names:
    
    current_model = init_model(model_name)
    print("--------------------------")
    print("Predictions vom Modell {} ".format(model_name))
    print("Top 1 - Prediction: {}".format(predict_image(current_model,jpgfile)))
    predictions_top_image = predict_top_image(current_model,jpgfile,top_value=TOP_VALUE)
    headers=['Class name', 'index','prediction']
    table = []
    # Ausgabe der n ersten erkannten Klassen
    for i in range (0,TOP_VALUE):
        class_index = predictions_top_image[0][i]
        table.append([str(get_imagenet_class(class_index)),class_index, predictions_top_image[1][i]])
    print(tabulate(table,headers = headers,tablefmt='orgtbl'))
    print("--------------------------")