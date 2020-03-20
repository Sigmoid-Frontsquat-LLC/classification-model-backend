import sys # this is for extracting command line arguments.
import pandas as pd
import io
import requests
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt








def parse_activator(flag, value):
    if flag[1] == 'a':
        return (True, value)
    else:
        return (False,None)
    pass

def parse_optimizer(flag, value):
    if flag[1] == 'o':
        return (True, value)
    else:
        return (False,None)
    pass
def parse_source(flag, value):
    if flag[1] == 's':
        return (True, value)
    else:
        return (False,None)
    pass

activator = ''
optimizer = ''
source = ''

if len(sys.argv) == 1 or (len(sys.argv) - 1) % 2 != 0:
    raise ValueError("Usage: -source [-a activator] [-o optimizer]")
else:
    # could this be done better?
    # sure, but this works for now...
    for i in range(1, len(sys.argv) - 1):
        flag = sys.argv[i]
        value = sys.argv[i + 1]

        isActivator, act = parse_activator(flag, value)

        if isActivator:
            activator = act
            continue

        isOptimizer, opt = parse_optimizer(flag, value)

        if isOptimizer:
            optimizer = opt
            continue
        isSource, so = parse_source(flag, value)

        if isSource:
            source = so
            continue
        pass
    pass

# naive check to ensure no argument is left unfilled
if len(activator) == 0 or len(optimizer) == 0 or len(source) or 0:
    raise ValueError("Usage: -source [-a activator] [-o optimizer]")

print("Hello, World!")
## each argument has been passed in, however
## we need to make sure they are valid.


#########################
# austinwilson
# robertocampos
# kevinpacheco
#
#
# this is my code buddy 




## basic model with terrible f1... 
# fix later just try to get simple version working 
# my image is 224,224,4 so i will use that dimension for now
# no idea why its 4..
num_classes = 10
input_shape = (224,224,3)






model = Sequential()
# was initially 32 kernel_size=(4,4)
# number of filters is 32
model.add(Conv2D(32,kernel_size=(3,3),strides = (1,1), padding='valid',
                activation = 'sigmoid',
                input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax'))


# need pre trained weights 
# model.compile(loss='categorical_crossentropy',optimizer='adam')
# model.load_weights('best_weights.hdf5')




## robertos modelo
# input_shape=(32, 32, 3)

modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation=activator, padding='same', input_shape=(32, 32, 3)))
modelo.add(Conv2D(32, (3, 3), activation=activator, padding='same'))
modelo.add(Conv2D(32, (3, 3), activation=activator, padding='same'))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Dropout(0.2))
modelo.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
modelo.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
modelo.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Dropout(0.2))
modelo.add(Conv2D(128, (3, 3), activation=activator, padding='same'))
modelo.add(Conv2D(128, (3, 3), activation=activator, padding='same'))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Flatten())
modelo.add(Dense(128, activation=activator))
modelo.add(Dropout(0.2))
modelo.add(Dense(10, activation='softmax'))


modelo.compile(loss='categorical_crossentropy',optimizer=optimizer)
modelo.load_weights('best_weights.hdf5')




## this is for transfer learning model


vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))   #  first hidden layer

 
# write your code here
model = Sequential()
for layer in vgg_model.layers:
  model.add(layer)

for layer in model.layers:
  layer.trainable = False









######################### not sure here

# validate the 'activator'
pass
# validate the 'optimizer'
pass

# Load weights based on activator and optimizer

# Preprocess the image information

# Get the classification

# Print out the classification