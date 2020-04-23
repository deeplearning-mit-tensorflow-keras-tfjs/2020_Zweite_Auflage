import tensorflow as tf
import numpy as np


# Initialisierung der Variablen 
tf.summary.trace_on(graph=True, profiler=True)
tf.config.experimental_run_functions_eagerly(True)

a = tf.Variable(3.0)
b = tf.Variable(4.0)
c = tf.Variable(1.0)
d = tf.Variable(2.0)

x_1 = tf.multiply(a, b, "Multiplikation")
x_2 = tf.add(c, d, "Addieren")
x_3 = tf.subtract(x_1, x_2, "Subtrahieren")
result = tf.sqrt(x_3, "Wurzel")

writer = tf.summary.create_file_writer("./logs")
with writer.as_default():
  
    tf.summary.trace_export(
      name="mein_Graph",
      step=100,
      profiler_outdir="logs")


print("Ergebnis der Berechnung: {}".format(result.numpy()))
