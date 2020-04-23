import tensorflow as tf
import numpy as np

tf.summary.trace_on(graph=True,profiler=True)

# Initialisierung der Variablen 
a = tf.Variable(3.0, name="Variable_a")
b = tf.Variable(4.0, name="Variable_b")
c = tf.Variable(1.0, name="Variable_c")
d = tf.Variable(2.0, name="Variable_d")

@tf.function
def my_function(a,b,c,d):
    x_1 = tf.multiply(a, b, "Multiplikation")
    x_2 = tf.add(c, d, "Addieren")
    x_3 = tf.subtract(x_1, x_2, "Subtrahieren")
    result = tf.sqrt(x_3, "Wurzel")
    return result;

with tf.device("/device:GPU:0"):
    res = my_function(a,b,c,d)

print(tf.autograph.to_code(my_function.python_function))

writer = tf.summary.create_file_writer("./logs")
with writer.as_default():
    tf.summary.trace_export(
      name="export_prfil",
      step = 0,
      profiler_outdir="logs")

print("Ergebnis der Berechnung des Graphen:{}".format(res.numpy()))