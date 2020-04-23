#
# Überprüfung des Inhaltes des IMDB-Dataset
#

from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, LSTM, Flatten, Dropout
from tensorflow.keras.preprocessing import sequence 
import numpy as np
import string

# Konstanten
VOCABULARY_SIZE = 88000 #20000
INDEX_FROM = 3
END_CHAR = 2
START_CHAR = 1
PAD_MAX_LENGTH = 1000 # 200
EMOJIS = ["👎","👍"] # 0 = Negativ 1 = Positive Bewertung

# Trainings und Testdaten werden über Keras geladen
# Alternativ können Sie direkt die Datei als Pikle Datei herunterladen
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
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
for review_index in range(0,50):
    print("---------- {} --------".format(str(review_index)))
    print("---- Rezensionstext --------- ")
    print(' '.join(id_to_word[id] for id in x_train[review_index] ))
    print("---- Label / Stimmung --------- ")
    print(EMOJIS[y_train[review_index]])