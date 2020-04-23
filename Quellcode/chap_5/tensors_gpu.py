#
# Explizite Benutzung der GPU in TensorFlow 2
#

import tensorflow as tf

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

def get_GPU_CPU_details():
    print("GPU Vorhanden? ", tf.test.is_gpu_available())
    print("Devices: ",tf.config.experimental.list_physical_devices())

get_GPU_CPU_details()

tf.test.gpu_device_name()

print(tf.config.experimental.list_physical_devices())
tf.test.gpu_device_name()

with tf.device("/device:GPU:0"):
    a = tf.constant([[0.0, 1.0, 2],[3,0,1]])
    b = tf.constant([[1.0,2.0],[4,6],[1,2]])
    result = tf.matmul(a, b)
    print(result)

with tf.device("/device:GPU:0"):
    d = tf.Variable(2.0)
    tf.print(d)


with tf.device("/device:CPU:0"):
    c = tf.constant([[0.0, 1.0, 2],[3,0,1]])
    d = tf.constant([[1.0,2.0],[4,6],[1,2]])
    res = tf.matmul(c, d)