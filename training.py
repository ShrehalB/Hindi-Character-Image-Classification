import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
import numpy as np
import cv2
import os
from sklearn.cross_validation import train_test_split
import glob
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json

folder = "padded/"

x = np.zeros((1966, 192, 218))
y = np.zeros((1966, 128))
i=0;

for filename in os.listdir(folder):
	name = filename.split(".")[0]

	tokens = name.split("_")

	tokens.pop(0)
	tokens.pop(0)
	tokens.pop(0)

	x[i] = cv2.imread(os.path.join(folder, filename), 0)

	while(len(tokens)!=0):
		y[i][int(tokens.pop(0))-2304]=1

	i=i+1
print(str(i) + " files loaded.\n\n")

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.33,random_state=55)

img_rows, img_cols = 192,218
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

img_rows, img_cols = 192,218
img_channels = 1
batch_size = 32
nb_classes = 128
nb_epoch = 10
nb_filters = 60
nb_pool = 2
nb_conv = 3

model = Sequential()

model.add(Convolution2D(60,kernel_size=(5,5),strides = (1,1),activation = 'relu',input_shape =(img_rows, img_cols,1)))
model.add(MaxPooling2D(pool_size=(5,5),strides=(2,2)))
model.add(Dropout(0.15))
model.add(Convolution2D(30,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3,3)))
model.add(Dropout(0.25))
model.add(Convolution2D(15,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(1000,activation='tanh'))
model.add(Dense(128,activation='softmax'))
model.compile(loss=keras.losses.binary_crossentropy,optimizer = keras.optimizers.Adam(),metrics = ['categorical_accuracy'])

model.fit(x_train,y_train,batch_size =32,epochs =5,verbose = 1,validation_data = (x_test,y_test))

print("Saving model...")
model.save_weights("./models/1_softmax_weight.h5")
model.save('./models/1_softmax_model.h5')
print("Model saved to disk.")