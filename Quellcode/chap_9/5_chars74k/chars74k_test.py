#
# Test des Modells Chars74k
# 

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random
import numpy as np
import string
import matplotlib
import matplotlib.pyplot as plt
import gzip
from prettytable import PrettyTable
from skimage import color, exposure, transform, io
from PIL import Image
from skimage.io import imread
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import progressbar

bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

# mit Augemented 
# Evaluation / Loss 0.05943059536534141, Acc:0.9827901721000671

# Auf data/test Dataset
# Anzahl Dateien: 5080
# Korrekte Matches: 3824
# Trefferquote:75.27559055118111


# mit Bmp
# Evaluation / Loss 1.0019335079935756, Acc:0.792991578578949
# Auf data/test Dataset
# Anzahl Dateien: 5080
# Korrekte Matches: 3790
# Trefferquote:74.60629921259843



TEST_DIRECTORY = "data/test/"

IMAGE_SIZE = 28; 

labels_decoded = string.digits + string.ascii_uppercase + string.ascii_lowercase

# Wir laden unser Modell 
model = load_model("chars74.h5")

# Lädt ein Bild
def load_image(path,image_size):
    img = cv2.imread(path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,image_size,None,interpolation=cv2.INTER_CUBIC)
    img = np.array(img,dtype="float32")
    img /=255
    img = img.reshape([-1,image_size[0],image_size[1],1])
    return img

# Test auf Dataset
def test_on_dataset(path):
    correct_matches = 0
    file_count = 0
    result_table = PrettyTable()
    result_table.field_names = ["Datei ", "Ist", "Dekodiert", "Match?"]

    for directories in sorted(os.listdir(path)):
        if(directories.startswith('.') == False):
            for filename in sorted(os.listdir(path + directories)):
        
                    if(filename.startswith('.') == False):
                        file_count = file_count +1

                        current_letter = filename.replace("img","").replace(".png","")
                        current_letter = int(current_letter[0:current_letter.find("-")])
                        current_letter = labels_decoded[current_letter-1]

                        image_path = path +directories+"/"+filename
                       
                        test_image = load_image(image_path,(IMAGE_SIZE,IMAGE_SIZE))
                        predictions = model.predict(test_image)
                        index_max_predictions = np.argmax(predictions)
                        decode_letter = labels_decoded[index_max_predictions]

                        # Passt oder nicht?
                        if( str.upper(current_letter) == str.upper(decode_letter)):
                            result_table.add_row([image_path, current_letter, decode_letter, "✅" ])
                            correct_matches = correct_matches + 1 
                        else:
                            result_table.add_row([image_path, current_letter, decode_letter, "❌" ])

                        bar.update(file_count)
                        if(file_count%100==0):
                            bar.update(file_count)
                            print("Lese Datei: {}".format(file_count))

    print(result_table)
    print("Anzahl Dateien: {}".format(file_count))
    print("Korrekte Matches: {}".format(correct_matches))
    print("Trefferquote:{}".format(correct_matches/file_count * 100))

test_on_dataset(TEST_DIRECTORY)