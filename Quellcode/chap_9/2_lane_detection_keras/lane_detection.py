#
# Projekt 2: Spurerkennung mit Keras
#
# Hinweis: Bug in Keras von TensorFlow 2 in 
# engine/saving.py beim Laden eines Modells
# verursacht ein KeyError: 'weighted_metrics'
# from tensorflow.python.keras.models import load_model
# Aus dem Grund wurde hier die Keras.io (Keras-2.3.1) Version genommen

import keras
import numpy as np
import cv2

from keras.models import load_model
from scipy.misc import imresize

# Laden des Keras Modells
model = load_model('full_CNN_model.h5')
w = model.load_weights('full_CNN_model.h5')

print(w)
model.summary()

print(model.to_json())

lane_recent_fit = []
lane_avg_fit = []

# Diese Funktion nimmt ein Bild als Eingabeparameter und 
# zeichnet über das Bild die Zonen in denen die Spur erkannt wurde
def road_lines_image(imageIn):

    # Das Bild wird auf eine Größe von 160 x 80 verkleinert 
    small_img = imresize(imageIn, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Vorhersage vom Model
    prediction = model.predict(small_img)[0] * 255

    # Hier wird die Vorhersage des Keras Netz visualisiert
    cv2.imshow("Keras Prediction",prediction)

    global lane_recent_fit
    global lane_avg_fit

    # Die Vorhersagen werden gespeichert... 
    lane_recent_fit.append(prediction)

    # ... die letzten 5 Vorhersagen werden genommen ...
    if len(lane_recent_fit) > 5:
        lane_recent_fit = lane_recent_fit[1:]

    # ... und den Mittelwert davon ermittelt
    lane_avg_fit = np.mean(np.array([i for i in lane_recent_fit]), axis = 0)
    
    # Hier werden zusätzlich die Farben R(ot) und B(blau) gesetzt, weil die Anzeige eines Bild mit CV2
    # einen Array mit RGB-Werte erwartet. 
    red_blanks = np.zeros_like(lane_avg_fit).astype(np.uint8)
    blue_blanks = np.zeros_like(lane_avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((red_blanks, lane_avg_fit, blue_blanks))

    # Die Größe des Bild der Spurzone und des Original werden zusammengefügt
    lane_image = imresize(lane_drawn, imageIn.shape)
    alpha = 0.5
    result = cv2.addWeighted(imageIn, alpha, lane_image, 1, 0) 
    return result

# Abspielen eines Videos z.B. einer Dashcam
cap = cv2.VideoCapture('dash_cam.mp4')
#cap = cv2.VideoCapture(0) # Kommentieren Sie die Zeile aus, wenn Sie das Video von der Webcamera kommen soll
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
    frame = road_lines_image(frame)
    cv2.imshow('Lane detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()