#
# Textausgabe im TensorBoard
# 	

import tensorflow as tf
import numpy as np

# Hinweis : vergessen Sie nicht TensorBoard mit folgender Zeile zu starten:
# tensorboard --logdir="./logs_text"
train_summary_writer = tf.summary.create_file_writer("logs_text")

with tf.name_scope('Meine_Textausgabe'):
        with train_summary_writer.as_default():
            # Als Text + Emoji
            tf.summary.text('Ausgabe 1"', "Hallo TensorFlow 😀",step=0)  
            # Als HTML
            tf.summary.text("Tabelle","<table><thead><tr><th>Eine Tabelle </th></thead><tbody><tr><td>Eintrag 1</td></tr><tr><td>Eintrag 2</td></tr></tbody></table>",step=1)  
            # Als HTML-Link
            tf.summary.text("Link","<a href=\"http://www.tensorflow.org\">Link</a>",step=2)  
            # Als HTML-Liste
            tf.summary.text("Liste","<ol><li>Item 1</li><li>Item 2</li></ol>",step=3)  
        