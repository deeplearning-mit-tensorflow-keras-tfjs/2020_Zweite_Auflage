#
#  Einfaches CNN für Fashion-MNIST mit TensorFlow
#

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 



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

# Die Daten werden in ein 4-D Tensor umgewandelt mit einer Breite von 
# 28 Pixel, einer Höhe von 28 Pixel und einer Farbtiefe von 1,  
# da wir Graustufen als Farbkanal verwenden   
train_data = train_data.reshape(-1, 28, 28, 1)

# Placeholders für die Bilder und die Labels
images = tf.placeholder("float", [None, 28,28,1],"images")
labels = tf.placeholder("float", [None, 10],"labels")


#  Definition des Modells #
def build_fashion_model(input):

  # Input Layer
  input_layer = tf.reshape(images, [-1, 28, 28, 1])
 
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters= 32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
  # Convolutional Layer #2  
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  #Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Wird auf eine Dimension reduziert, damit es als Eingabe für den dense Layer benutzt werden kann 
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

   # Dense Layer bzw. Fully connected
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Dropout ()
  dropout = tf.layers.dropout(inputs=dense, rate=0.4)

  # Fully connected
  # Das Netz besitzt 10 Ausgabenklassen (units=10)
  # Hier werden die logits bestimmt. In der loss_op Operation werden
  # die Vorhersagen dank der Softmax-Funktion ermittelt

  logits = tf.layers.dense(inputs=dropout, units=10)

  return logits

# Graphoperationen
pred = build_fashion_model(images)

#correct_predictions_op = np.equal(np.argmax(pred), np.argmax(labels))
correct_predictions_op = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))

accuracy_op = tf.reduce_mean(tf.cast(correct_predictions_op, tf.float32))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=labels))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op) 

# Variablen für das Training
EPOCHS_NUM = 100
epochs = range(0,EPOCHS_NUM)
BATCH_SIZE = 64


# Beachten Sie jedoch, dass BATCH_SIZE<train_labels
batches = range(0,len(train_labels)//BATCH_SIZE)

# Hilfsfunktion, um Daten aus dem Batch von index_start bis index_end zu laden
def get_next_batch(data,batch):
    batch_start_index = batch * BATCH_SIZE
    batch_end_index = min((batch+1)*BATCH_SIZE, len(data))
    return data[batch_start_index:batch_end_index]



config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Der Graph wird initialisiert: 
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in epochs:
        for b in batches:
           
            # Batch für die Features
            img_input = get_next_batch(train_data,b)
            
            # Batch für die Labels
            labels_output = get_next_batch(train_labels,b)
            
            sess.run(train_op,feed_dict={images:img_input,labels:labels_output})
            
            loss, acc = sess.run([loss_op,accuracy_op],feed_dict={images:img_input,labels:labels_output})

        print("Epochs: {} Accuracy: {} Loss:{}".format(i,acc,loss))

    # Schnell test
    input_picture = eval_data[0].reshape(28,28)
    input_label = fashion_class_labels[np.argmax(eval_labels[0], axis=None, out=None)]
    print ("Selected: {}".format(input_label))
    plt.title(input_label)
    plt.imshow(input_picture,cmap='Greys')
    print(input_picture.shape)

    # my_test_image = tf.reshape(input_picture, [-1, 28, 28, 1])
    predictions = sess.run(pred,feed_dict={images:[[input_picture]]})
    index = int(np.argmax(predictions,axis=1))
        
    # Vorhersage:
    print("Gefundene Fashion Kategorie: {}".format(fashion_class_labels[index]))
    
