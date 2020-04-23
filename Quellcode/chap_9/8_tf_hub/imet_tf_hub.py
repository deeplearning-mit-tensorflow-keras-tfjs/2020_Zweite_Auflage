import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.keras.backend import eval
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pprint
pp = pprint.PrettyPrinter()

from operator import itemgetter

# Siehe https://github.com/tensorflow/hub/issues/350
tf.disable_eager_execution()

def show_top_classes(logits, top_classes):
    class_indexes = np.argpartition(logits, -top_classes)[-top_classes:]
    pred = np.array(logits[class_indexes])
    all_scores = []
    for i in class_indexes:
        all_scores.append([met_labels[i],logits[i]])

    all_scores.sort(key=lambda tup: tup[1])
    all_scores = all_scores[::-1]
    return all_scores

met_labels = np.genfromtxt("imetv1_labelmap.csv",delimiter=',',dtype='str',usecols=[1],skip_header=True)

module = hub.Module("https://tfhub.dev/metmuseum/vision/classifier/imet_attributes_V1/1")
print("Eingabedimension vom Bild: H:{} px X W:{} px ".format(hub.get_expected_image_size(module)[0],hub.get_expected_image_size(module)[1]))

# Hier bekommen wir zusätzliche Informationen über das Module von TensorFlow Hub
# Eingabe
print(module.get_input_info_dict()) 
# Ausgabe
print(module.get_output_info_dict())

# Eingabebilder 
input_image = plt.imread("DT11140.jpg")
input_image = input_image.astype(np.float32)[np.newaxis, ...] / 255.
input_image = tf.image.resize(input_image, (299, 299))

logits = eval(module(input_image)[0])

pp.pprint(show_top_classes(logits,5))
