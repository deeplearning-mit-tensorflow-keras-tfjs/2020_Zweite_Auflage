#
#  Einfaches CNN für Fashion-MNIST mit TensorFlow 2.x
#

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import gzip
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

'''
def show_dataset():
    plt.figure(figsize=(5,5))
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title(fashion_class_labels[train_labels[i]])
        plt.imshow( np.reshape(train_images[i],(28,28)),cmap='Greys')

    plt.show()
'''


show_test_image(2500)
#show_dataset()


print(test_labels[2500])
#print(test_labels)


def fashion_model():

    # Ein einfaches Model 
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


# Ohne one_hot encoding:
# model.compile(optimizer="adam",loss= tf.losses.sparse_categorical_crossentropy,metrics=["accuracy"])

model = fashion_model()
model.compile(optimizer="adam",loss= tf.keras.losses.categorical_crossentropy,metrics=["accuracy"])

# Mit TensorBoard Visualisierung
# tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
# model.fit(train_images,train_labels,epochs=10,callbacks=[tf_callback])

model.fit(train_images,train_labels,epochs=100,batch_size=32, validation_split=0.1)

evaluation_results = model.evaluate(test_images, test_labels)
print(evaluation_results)
print("Loss: {}".format(evaluation_results[0]))
print("Accuracy: {}".format(evaluation_results[1]))

# Modell wird gespeichert und neu geladen
model.save("fashion.h5")
fashion_model = load_model("fashion.h5")

# Laden von Bild 120 und Vorhersage mit keras
predictions = fashion_model.predict(get_test_image(2500))
        
# Vorhersage:
print("Gefundene Fashion Kategorie: {}".format(fashion_class_labels[np.argmax(predictions)]))

# Alternative schreibweise
'''
model = tf.keras.models.Sequential([    
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(60,activation="relu"),
    tf.keras.layers.Dense(20,activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
'''