import tensorflow as tf
import cv2
import numpy as np
import requests
import urllib

from tensorflow import keras
from PIL import Image

from tf_explain.core.grad_cam import GradCAM

# Das Bild wird für das Modell auf 224x224 Pixel reduziert 
# und die Pixelwerte normalisiert
test_image = Image.open("cat.jpg")
test_image = test_image.resize((224,224), Image.ANTIALIAS)
test_image = np.array(test_image,dtype="float32")
test_image /= 255
test_image = test_image.reshape(224,224,3)

# Vorhersage von einem Modell
# model = tf.keras.models.load_model('cifar_model.h5')

model = tf.keras.applications.VGG16(include_top=True,weights="imagenet")
model.summary()
img = tf.keras.preprocessing.image.load_img("cat.jpg", target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
data = ([test_image], None)
explainer = GradCAM()

tabby_cat_class_index = 281


# Den class_index findet man innerhalb der imagenet_class_index.json Datei
# siehe https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
# Die nummer 867 => trailer_truck
# 717 => pickup
# 301 => ladybug
grid = explainer.explain(data, model, class_index=tabby_cat_class_index,layer_name="block5_conv3")
explainer.save(grid, ".", "grad_cam.png")