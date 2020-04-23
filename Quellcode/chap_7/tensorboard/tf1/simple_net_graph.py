# 
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 

from tensorboard import version; print(version.VERSION)

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
      activation=tf.nn.relu,name="Zweites_Conv2D")

  #Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Wird flach gemacht, damit es als Eingabe für den dense Layer benutzt werden kann 
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
   # Dense Layer bzw. Fully connected
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  # Dropout
  dropout = tf.layers.dropout(inputs=dense, rate=0.4)
  # Logits mit units=10, da wir 10 Ausgabenklassen haben
  logits = tf.layers.dense(inputs=dropout, units=10)

  return logits

# Operation
pred = build_fashion_model(images)
correct_prediction_op = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float32))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=labels))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op) 

# Variablen für das Training
EPOCHS_NUM = 10
epochs = range(0,EPOCHS_NUM)
BATCH_SIZE = 64
batches = range(0,len(train_labels)//BATCH_SIZE)

# Hilfe Funktion, um Daten aus dem von index_start bis index_end
def get_next_batch(data,batch):
    batch_start_index = batch * BATCH_SIZE
    batch_end_index = min((batch+1)*BATCH_SIZE, len(data))
    return data[batch_start_index:batch_end_index]



fashion_summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph(),filename_suffix="fashion")
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# Der Graph wird initialisiert 
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    steps = 0
    for i in epochs:
        for b in batches:
           
            # Batch für die Features
            img_input = get_next_batch(train_data,b)
            
            # Batch für die Labels
            labels_output = get_next_batch(train_labels,b)
            
            sess.run(train_op,feed_dict={images:img_input,labels:labels_output},options=run_options,run_metadata=run_metadata)
            
            #loss, acc = sess.run([loss_op,accuracy_op],feed_dict={images:img_input,labels:labels_output})

            steps = steps + 1 
            fashion_summary_writer.add_run_metadata(run_metadata,tag="session_{}".format(steps))

        #print("Epochs: {} Accuracy: {} Loss:{}".format(i,acc,loss))
