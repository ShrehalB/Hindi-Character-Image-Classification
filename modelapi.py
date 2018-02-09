import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model
import os

# folder="./shrehal_train/"
model = load_model('deep_model.h5')

def preproc(image):
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5,5),np.float32)/25
    image = cv2.filter2D(image,-1,kernel)
    ret, ret_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    res_image = cv2.resize(ret_image, (64,64))
    
    return res_image

def predict(filename):
    # image = cv2.imread(filename, 0)
    image = cv2.imread(filename, 0)
    image = preproc(image)
    image = image.astype('float32')
    image = image/255.0
    image = image.reshape((1,64,64,1))
    pred_array = model.predict(image)
    pred_array[pred_array < 0.50] = 0
    pred_array[pred_array >= 0.50] = 1
    
    result = []
    for predic in range(len(pred_array[0])):
        if pred_array[0][predic] == 0:
            continue
        
        res = predic + 2304
        result.append(res)
    
    return result

# i=1;
# for filename in os.listdir(folder):
#     print(filename)
#     print(predict(filename))
