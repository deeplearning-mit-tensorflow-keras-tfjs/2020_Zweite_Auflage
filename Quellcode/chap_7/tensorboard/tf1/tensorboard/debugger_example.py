#
# Beispiel Benutzung des TensorBoard Debuggers
# 

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 
from tf_cnnvis import *
from tensorflow.python import debug as tf_debug 

# Fashion Klassen
fashion_class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Labels und Daten werden hier geladen
def load_fashion_data():
    data = input_data.read_data_sets('data/fashion',one_hot=True)

    # Trainingsdaten
    train_data = data.train.images
    train_labels = data.train.labels
    train_labels = np.asarray(data.train.labels,dtype=np.int32)
    
    # Evaluationsdaten
    eval_data = data.test.images  
    eval_labels = data.test.labels
    eval_labels = np.asarray(data.test.labels,dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    return (train_data, train_labels, eval_data, eval_labels) 

# Laden der Daten
train_data, train_labels, eval_data, eval_labels = load_fashion_data()
train_data = train_data.reshape(-1, 28, 28, 1)

# Placeholders für die Bilder und die Labels
images = tf.placeholder("float", [None, 28,28,1],"images")
labels = tf.placeholder("float", [None, 10],"labels")

####  Definition des Modells ###
def build_fashion_model(input, labels):

  # Input Layer
  input_layer = tf.reshape(images, [-1, 28, 28, 1])
 
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters= 32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu, name="1_Conv2D")

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name="2_MaxPooling")
    
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,name="3_Conv2D")

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,name="4_MaxPooling")

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,name="5_FullyConnected")
  dropout = tf.layers.dropout(inputs=dense, rate=0.4,name="6_DropOut")
  output_layer = tf.layers.dense(inputs=dropout, units=10,name="7_OutputLayer")

  return output_layer, [input_layer, conv1, pool1, conv2, pool2, dense, output_layer]


pred, layers = build_fashion_model(images,labels)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=labels))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op) 

EPOCHS_NUM = 10
epochs = range(0,EPOCHS_NUM)

BATCH_SIZE = 64 # Zum testen: Wenn Sie nur ein Bild im Beholder sehen möchten: BATCH_SIZE = 1
batches = range(0,len(train_labels)//BATCH_SIZE)

# Liefert den nächsten Batch zurück
def get_next_batch(data,batch):
    batch_start_index = batch * BATCH_SIZE
    batch_end_index = min((batch+1)*BATCH_SIZE, len(data))
    return data[batch_start_index:batch_end_index]

with tf.Session() as sess:
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:12345")
    sess.run(tf.global_variables_initializer())
    for i in epochs:
        for b in batches:
            # Batch für die Features
            img_input = get_next_batch(train_data,b)
            # Batch für die Labels
            labels_output = get_next_batch(train_labels,b)
            _,loss,acc = sess.run([train_op,loss_op,accuracy_op], feed_dict={images:img_input,labels:labels_output})