#
#  Einfaches CNN für Fashion-MNIST mit TensorFlow 2.x
#

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import gzip

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Input
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model

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
    
    # Trainingsdaten
    ## Labels (Ausgabe)
    with gzip.open("data/fashion/train-labels-idx1-ubyte.gz") as path:
        train_labels = np.frombuffer(path.read(),np.uint8,offset=8)

    ## Bilder (Eingabe)
    with gzip.open("data/fashion/train-images-idx3-ubyte.gz") as path:
        train_images = np.frombuffer(path.read(),np.uint8,offset=16).reshape(len(train_labels),28,28)/255.0

    # Testdaten 

    ## Labels (Ausgabe)
    with gzip.open("data/fashion/t10k-labels-idx1-ubyte.gz") as path:
        test_labels = np.frombuffer(path.read(),np.uint8,offset=8)

    ## Bilder (Eingabe)
    with gzip.open("data/fashion/t10k-images-idx3-ubyte.gz") as path:
        test_images = np.frombuffer(path.read(),np.uint8,offset=16).reshape(len(test_labels),28,28)/255.0

    test_images = np.reshape(test_images,(-1, 28, 28, 1))
    train_images = np.reshape(train_images,(-1, 28, 28, 1))


    seed = 42 
    # Shuffle der Testdaten
    train_images, train_labels = shuffle(train_images, train_labels,random_state=seed)

    # Shuffle der Testdaten
    test_images, test_labels = shuffle(test_images, test_labels,random_state=seed)

    # One Hot Encoding (10 Kategorien)
    train_labels = tf.keras.utils.to_categorical(train_labels,10) 
    test_labels = tf.keras.utils.to_categorical(test_labels,10)

    return train_images, train_labels, test_images, test_labels


# Laden der Daten
train_images, train_labels, test_images, test_labels = load_fashion_data()


def get_test_image(index):
    return np.reshape(test_images[index],(-1,28,28,1))


def show_test_image(index):
    plt.title(fashion_class_labels[np.argmax(test_labels[index])])
    plt.imshow(np.reshape(test_images[index],(28,28)),cmap='Greys')
    plt.show()

 # Ein einfaches Model 
def fashion_model():

    model = tf.keras.models.Sequential()

    # Convolutional Layer #1  
    model.add(tf.keras.layers.Conv2D(filters= 32,kernel_size=[5, 5], activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))

    # Convolutional Layer #2  
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=[5, 5], activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model


model = fashion_model()

train_summary_writer = tf.summary.create_file_writer("logs")

# Definition eines neuen Callbacks
class MyTensorBoardCallback(keras.callbacks.Callback):

    # Wir wollen am Ende jedes Epochs drei tf.summary rausschreiben
    def on_epoch_end(self,epoch,logs=None):
        with train_summary_writer.as_default():
            with tf.name_scope('Meine Metriken'):
                tf.summary.scalar('Aktueller Loss-Wert', logs["loss"],step=epoch,description="Der aktuelle Loss vom Modell")
                tf.summary.scalar('Aktueller Accuracy-Wert', logs["accuracy"],step=epoch,description="Die aktuelle Genauigkeit vom Modell")
                tf.summary.scalar('Epoche', epoch ,step=epoch,description="Die aktuelle Epoche während des Trainings")

# Eigenes Callback
my_tfb_cb = MyTensorBoardCallback()


with train_summary_writer.as_default():
    tf.summary.scalar("Wert:", 2 ,step=1,description="Ein Beschreibungstext ")
    # Beispiel Histograms-Dashboard
    with tf.name_scope('Histogram'):
        for i in range (0,20):
            tf.summary.histogram("Wert:",tf.convert_to_tensor([i]),step=i,description="Beispiel für Histogram-Dashboard")



model.compile(optimizer="adam",loss= tf.keras.losses.categorical_crossentropy,metrics=["accuracy"])
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="logs",histogram_freq=1)


model.fit(train_images,train_labels,epochs=10,batch_size=32, callbacks=[tf_callback], validation_split=0.1)

evaluation_results = model.evaluate(test_images, test_labels)
print(evaluation_results)
print("Loss: {}".format(evaluation_results[0]))
print("Accuracy: {}".format(evaluation_results[1]))

# Modell wird gespeichert und neu geladen
model.save("fashion.h5")
fashion_model = load_model("fashion.h5")