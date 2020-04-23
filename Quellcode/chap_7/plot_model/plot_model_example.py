# 
# Verwendung der Funktion plot_model() in Keras zur statischen Visualisierung eines Modelles
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
model = VGG16()
# Als PNG
tf.keras.utils.plot_model(model, to_file='model_output.png', show_shapes=True, show_layer_names=True,rankdir="TB")
# Als SVG
tf.keras.utils.plot_model(model, to_file='model_output.svg', show_shapes=False, show_layer_names=True,rankdir="TB")