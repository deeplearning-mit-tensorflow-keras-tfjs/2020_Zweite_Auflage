#
#  Benutzung des Modells (KITTI-Dataset)
#  Originalcode: https://github.com/6ixNugget/Multinet-Road-Segmentation
#  adaptiert für neuere Keras Versionen
#

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.misc

from skimage import color, exposure, transform, io
from tensorflow.keras.models import load_model, Model

IMAGE_SHAPE = (160, 576)

model = load_model("kitti_road_model.h5")

def get_lane(image):
    
    im_softmax = model.predict(np.array([image]))
    im_softmax = im_softmax[0][:, :, 1].reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    segmentation = (im_softmax > 0.5).reshape(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    lane = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    return lane

cap = cv2.VideoCapture("dash_cam.mp4") 

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(960,540))

    # Hier wird der Videobereich mit der Autobahnspur extrahiert, weil 
    # das Dashcam Video ein Teil des Armaturenbretts beinhalten
    frame = frame[0:420, 0:960] 
    frame = cv2.resize(frame, (576,160))

    lane = get_lane(frame)
    lane = cv2.blur(lane, (5, 5))
    combined = cv2.addWeighted(frame,1,lane,0.5,0)
    combined = cv2.resize(combined, (800,450))
    
    cv2.imshow('Lane detection',combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break