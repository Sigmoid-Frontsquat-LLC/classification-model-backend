import sys # this is for extracting command line arguments.

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
    raise ValueError("Usage: [-s image] [-a activator] [-o optimizer]")
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
if len(activator) == 0 or len(optimizer) == 0 or len(source) == 0 :
    raise ValueError("Usage: [-s image] [-a activator] [-o optimizer]")

print('Source: ' + source)
print('Activator: ' + activator)
print('Optimizer: ' + optimizer)

exit(0)

############# Classification Logic ##################

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

# airplane : 0
# automobile : 1
# bird : 2
# cat : 3
# deer : 4
# dog : 5
# frog : 6
# horse : 7
# ship : 8
# truck : 9

class_labels = ['airplane','automobile','bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = 10
# input_shape = (32,32,3)

# Image preprocessing
source = 'datcat.jpg'
img = Image.open(source)
img = img.resize((32, 32))
img_array = np.asarray(img)
img_array = img_array / 255
input_shape = (32,32,3) #<<<<<<<< is this supposed to be a 4?
# input_shape = img_array.shape

# reshape for model
# original model was trained with (32,32,3) 
# img_array = img_array.reshape((1,img_array.shape[0],img_array.shape[1],img_array.shape[3]))

img_array = img_array.reshape((1,32,32,3))

modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation=activator, padding='same', input_shape=input_shape))
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

# validate the 'activator'
pass
# validate the 'optimizer'
pass

# Load weights based on activator and optimizer
# probably not needed as we are already passing the optimizer as a variable 
modelo.compile(loss='categorical_crossentropy',optimizer=optimizer)

if optimizer == 'adam':
    # compile with adam 
    modelo.compile(loss='categorical_crossentropy',optimizer=optimizer)

    # activator 
    if activator == 'relu':
        # load adam-relu
        modelo.load_weights('dnn/relu-adam.hdf5')
    elif activator == 'sigmoid':
        # load sigmoid-adam
        modelo.load_weights('dnn/sigmoid-adam.hdf5')
    elif activator == 'tanh':
        # load tanh-adam
        modelo.load_weights('tanh-adam.hdf5')
    else:
        print('error')
elif optimizer == 'sgd':
    # compile with sgd
    if activator == 'relu':
        # load relu-sgd
        modelo.load_weights('relu-sgd.hdf5')
    elif activator == 'sigmoid':
        # load sigmoid-sgd
        modelo.load_weights('sigmoid-sgd.hdf5')
    elif activator == 'tanh':
        # load tanh-sgd
        modelo.load_weights('tanh-sgd.hdf5')

else: 
    print('error')

# Get the classification

# pred = modeo.predict()

# Print out the classification
print('dats a cat gro')