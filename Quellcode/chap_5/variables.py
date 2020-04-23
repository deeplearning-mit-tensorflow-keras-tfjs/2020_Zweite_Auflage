#
# Beispiel der Verwendung von tf.Variable() und tf.assign() in TensorFlow 2
#

import tensorflow as tf
import numpy as np

tf.debugging.set_log_device_placement(True)
tf.test.gpu_device_name()


my_tensor = tf.Variable([[1, 2, 3], [4,5,6]])

# Erstes Beispiel
t = np.array([2]) 

# Äquivalent zu my_tensor = my_tensor * 2 
my_tensor = my_tensor * t
my_tensor = tf.square(my_tensor) / 4
my_tensor = my_tensor - 1
print("Ausgabe Beispiel 1:")
print(my_tensor)
tf.print(my_tensor)

# Ausgabe:
# [[ 0.  3.  8.]
# [15. 24. 35.]]


# Zweites Beispiel:
my_tensor = tf.Variable([[1, 2, 3], [4,5,6]])
my_second_tensor = my_tensor.assign([[0,0,7],[0,0,8]])
print("Ausgabe Beispiel 2:")
tf.print(my_second_tensor)
    
# Ausgabe:
# [[0 0 7]
# [0 0 8]]

# Platzieren einer Variable auf die erste Verfügbare GPU
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

with tf.device("/device:GPU:0"):
  meine_variable = tf.Variable([1,2,3])
  meine_variable = meine_variable.assign([0,0,7])
  tf.print(meine_variable)