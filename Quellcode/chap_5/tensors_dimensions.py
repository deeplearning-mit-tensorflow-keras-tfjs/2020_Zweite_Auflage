#
# Benutzung von Tensoren mit TensorFlow 2
#

import tensorflow as tf
import numpy as np


my_scalar = tf.constant(1,name="mein_Skalar")
print(my_scalar) 
tf.print(my_scalar)

my_scalar = tf.Variable(1)

#Ausgabe: Tensor("Const_2:0", shape=(1, 1), dtype=int32)
tensor_0d = tf.constant(3)
print(tensor_0d)

tensor_1d = tf.constant([1,2,3,4])
print(tensor_1d)

tensor_1d_with_strings = tf.constant(["Hallo","Welt","dies","ist","ein 1D Tensor!"])
print(tensor_1d_with_strings)

tensor_2d_with_strings = tf.constant([["Petra","Schmitt"],["Max","Mustermann"],["John","Doe"]])
print(tensor_2d_with_strings)
tf.print(tensor_2d_with_strings)

tensor_2d_with_integer = tf.constant([[1,2],[3,4],[5,6]])
print(tensor_2d_with_integer)

tensor_3d = tf.constant([
                            [[1,4],[3,8]],
                            [[5,7],[9,3]]
                        ])
tf.print(tf.rank(tensor_3d))

# Beispiel numpy zu Tensor

np_array = np.arange(0,5,step=0.5)
print("Eine Numpy-Struktur")
tensor_from_numpy = tf.constant(np_array,name="converted",dtype=tf.float16)
print("... kann als Tensor umgewandelt werden:")
print(tensor_from_numpy)
# Ausgabe: 
print("... und wieder als Numpy-Struktur zurück:")
print(tensor_from_numpy.numpy())
# Ausgabe
# [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5]



