#
# Textausgabe im TensorBoard mit tf.summary
# 	

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import reuters

# Laden des Reuters-Datensatz
INDEX_FROM = 3
START_CHAR = 1
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=START_CHAR,
                                                         oov_char=2,
                                                         index_from=INDEX_FROM)

# Mapping Funktion von id auf Wort
word_index = reuters.get_word_index(path="reuters_word_index.json")
word_index = {k:(v+INDEX_FROM) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = START_CHAR # 1
word_index["<UNK>"] = 2
id_to_word = {value:key for key,value in word_index.items()}

# Funktion, die uns die Reuters Nachricht als String zurück gibt
def get_reuters_news(index):
    return ' '.join(id_to_word[id] for id in x_train[index] )
    
# Hinweis : vergessen Sie nicht TensorBoard mit folgender Zeile zu starten:
# tensorboard --logdir="./logs_reuters" 

train_summary_writer = tf.summary.create_file_writer("logs_reuters")

for i in range (0,10):
    news = get_reuters_news(i)
    with train_summary_writer.as_default():
        tf.summary.text('News', news,step=i,description="Reuters News")             