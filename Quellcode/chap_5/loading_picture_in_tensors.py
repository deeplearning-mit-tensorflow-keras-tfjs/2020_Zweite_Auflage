#
# Laden und Umkonvertierung eines RGB-Bildes zu einem Tensor mit Pillow und TensorFlow 2
#

import tensorflow as tf
import numpy as np
from PIL import Image


# Alternative 1: Mit Pillow
print ("== Alternative 1==")
img = Image.open("cat.jpg" )
img.load()
img_data = np.asarray( img, dtype="int32" )
img_tensor = tf.convert_to_tensor(img_data, dtype=tf.int32)
tf.print(img_tensor)
print("Rank des Tensors: {}".format(tf.rank(img_tensor)))


# Alternative 2: Mit tf.image.decode_jpeg()
print ("== Alternative 2 == ")
img = tf.image.decode_jpeg(tf.io.read_file("cat.jpg"))
print("Rank des Tensors: {}".format(tf.rank(img)))
print("Shape des Bildes: {}".format(tf.shape(img)))
print("RGB-Werte vom Pixel (x:0,y:0) :",img[0][0])

# Alternative 3: Mit tf.image.decode_image ohne Angaben von Bildformat()
print ("== Alternative 3 == ")
img = tf.image.decode_image(tf.io.read_file("cat.jpg"))
print("Rank des Tensors: {}".format(tf.rank(img)))
print("Shape des Bildes: {}".format(tf.shape(img)))
print("RGB-Werte vom Pixel (x:0,y:0) :",img[0][0])