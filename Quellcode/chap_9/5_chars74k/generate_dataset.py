import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import gzip
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from skimage import color, exposure, transform, io
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
from PIL import Image
import cv2
from scipy import ndimage

# https://github.com/fhennecker/chars74k/blob/master/preprocessing.py

GOOD_IMAGES_DIRECTORY = "data/GoodImg/Bmp/"
EXPORT_IMAGES_DIRECTORY = "data/Augmented/"
#GOOD_IMAGES_DIRECTORY = "data/chars74k-lite/" # loss: 15.7942 - accuracy: 0.9628
# Loss: 12.467052515174665
# Accuracy: 0.9627547264099121
IMAGE_SIZE = 28;

# Parameter für die Generierung der Bilder

ROTATION_MIN_VALUE = -10
ROTATION_MAX_VALUE = 10

SCALE_MIN_VALUE = 0.9
SCALE_MAX_VALUE = 1.1


# https://stackoverflow.com/questions/42272384/opencv-removal-of-noise-in-image 
#
def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)


def rotate_scale_image(img,angle,scale):
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D( (cols / 2, rows / 2), angle, scale)
    rotated_image = cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))
    return rotated_image


total_images = 0


def generate_versions_images(image,path,filename):

    # Normal copy
    # cv2.imwrite(path + "/" + filename+".png", image) 
    global total_images
    for angle in np.arange(ROTATION_MIN_VALUE,ROTATION_MAX_VALUE,step=1):
        for scale in np.arange(SCALE_MIN_VALUE,SCALE_MAX_VALUE,step=0.1):
            # Generated copy
            rotated_and_scaled_image = rotate_scale_image(image,angle,scale)
            cv2.imwrite(path + "/g_" + filename + "_r"+str(angle)+"_s"+str(scale)+".png", rotated_and_scaled_image) 
            total_images = total_images + 1 

    # print("Anzahl generierter Bilder: {}".format(total_images))
   

def generate_additional_images():
    for directory in sorted(os.listdir(GOOD_IMAGES_DIRECTORY)):
        if(directory.startswith('.') == False):
            os.mkdir(EXPORT_IMAGES_DIRECTORY + directory)
            for filename in sorted(os.listdir(GOOD_IMAGES_DIRECTORY + directory)):
                filename = filename.replace(".png","")
                if(filename.startswith('.') == False):
                    pictureFilePath = GOOD_IMAGES_DIRECTORY + directory + "/" + filename
                   
                    
                    image = cv2.imread(pictureFilePath+".png")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    (thresh, image)  = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                    
                    image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
                    image= adjust_gamma(image,2.0) # Generiere Bilder, die heller sind

                    # Wir filtern die Bilder bei denen die
                    if(250<cv2.countNonZero(image)):
                        generate_versions_images(image,EXPORT_IMAGES_DIRECTORY + directory,filename)
                    #else:
                    #    print ("Image is black")
                    
                    
os.mkdir("data/Augmented")
generate_additional_images()
print("Generated!")