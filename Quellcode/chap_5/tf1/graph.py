import tensorflow as tf
import numpy as np

# Initialisierung der Variablen 
a = tf.Variable(3.0, name="Variable_a")
b = tf.Variable(4.0, name="Variable_b")
c = tf.Variable(1.0, name="Variable_c")
d = tf.Variable(2.0, name="Variable_d")

x_1 = tf.multiply(a, b, "Multiplikation")
x_2 = tf.add(c, d, "Addieren")
x_3 = tf.subtract(x_1, x_2, "Subtrahieren")
result = tf.sqrt(x_3, "Wurzel")

print("Ergebnis der Berechnung des Graphen:", res)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    res = sess.run(result)
    print("Ergebnis der Berechnung des Graphen:", res)
