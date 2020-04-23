#
# Projekt 6: Sentiment Analyse mit Keras
#
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import string

from tensorflow import keras
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, LSTM, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

# Konstanten
VOCABULARY_SIZE = 88000 
INDEX_FROM = 3
END_CHAR = 2
START_CHAR = 1
PAD_MAX_LENGTH = 1000 
EMOJIS = ["👎","👍"] # 0 = Negativ 1 = Positive Bewertung

# Einfaches Modell
def build_sentiment_model():
    model = Sequential()
    model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim = 100))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    return model;

# Modell mit LSTM
def build_sentiment_model_with_lstm():
    model = Sequential()
    model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim = 100,input_length=PAD_MAX_LENGTH))
    model.add(LSTM(100)) 
    model.add(Dense(1,activation="sigmoid"))
    return model;

# Hilfmethode
def predict_sentiment(message):
    message = message.lower();
    # Macht alle Satzzeichen weg
    message=message.translate(str.maketrans('','',string.punctuation))
    tmp = []
    for word in message.split(" "):
        tmp.append(word_to_id[word])
    padded_message = sequence.pad_sequences([tmp], maxlen=PAD_MAX_LENGTH)
    sentiment_prediction  = my_model.predict(np.array(padded_message))
    return sentiment_prediction

# Trainings und Testdaten werden über Keras geladen
# Alternativ können Sie direkt die Datei als Pikle Datei herunterladen
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=VOCABULARY_SIZE,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=START_CHAR,
                                                      oov_char=2,
                                                      index_from=INDEX_FROM)

# Die Datei wird imdb_word_index.json heruntergeladen
word_to_id = imdb.get_word_index(path = "./imdb_word_index.json")

# Hier werden die korrekten Indizes mit dem passenden Wort gespeichert, da es eine Index-Verschiebung von +3 gibt (siehe Erklärung in 
# https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification)
# Aus: https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = START_CHAR # 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}

# Zeigt den Inhalt einer Rezension (bestimmt durch REVIEW_INDEX)
REVIEW_INDEX = 2
print(X_train[REVIEW_INDEX])
print("---- Rezensionstext --------- ")
print(' '.join(id_to_word[id] for id in X_train[REVIEW_INDEX] ))
print("---- Label / Stimmung --------- ")
print(EMOJIS[y_train[REVIEW_INDEX]])

# Eingabe werde gepaddet, damit alle Rezesionslängen gleich sind
X_train = sequence.pad_sequences(X_train,maxlen=PAD_MAX_LENGTH)
X_test = sequence.pad_sequences(X_test,maxlen=PAD_MAX_LENGTH)

# Das Model wird gebaut

model_number = 2

if model_number == 1:
    my_model = build_sentiment_model_with_lstm()
elif model_number == 2:
    my_model = build_sentiment_model()

my_model.summary();
my_model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

'''history = my_model.fit(X_train, y_train, 
    validation_data=(X_test,y_test), batch_size=64, epochs=20,verbose=1,validation_split=0.4)'''

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state = 113)
history = my_model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=512, epochs=20,verbose=1)
scores = my_model.evaluate(X_test, y_test, verbose=1)
print("Test accuracy:",scores[1])

# Das Model wird gespeichert
my_model.save("sentiment_model.h5")
# Und auch für TensorFlow.js!
tfjs.converters.save_keras_model(my_model, "./tfjs_sentiment_model")


# Kurzer Test, ob die Analyse auch funktioniert:
#message = "this movie was terrible and bad"
#message = "i really liked the movie and had fun"
#message = "the film was not good noisy background music and the dialogs were simply bad"
#message = "this was a bad movie with a lot of errors"
review = "a beautiful and wonderful movie with a lot of nice and hilarious scenes"
print(predict_sentiment(review))

review = "this movie was terrible and bad"
print(predict_sentiment(review))

review = "the film was not good noisy background music and the dialogs were simply bad"
print(predict_sentiment(review))


### Bonus
import matplotlib.pyplot as plt 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Sentiment Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
results = my_model.evaluate(X_train,y_train)
print(results)
plt.show()
