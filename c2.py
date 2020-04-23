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
            if act != '-o':
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


# exit(0)

############# Classification Logic ##################

import pandas as pd
import io
import requests
import numpy as np
import os
import logging
import json
import shutil
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image, ImageFile, ImageEnhance
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


####### warning messages not printed #######
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class_labels = ['coupe','motorcycle','sedan','suv','truck']

num_classes = 10

# Image preprocessing

img = Image.open(source)
img = img.resize((256,256))
enhancer = ImageEnhance.Sharpness(img)
enhanced_im = enhancer.enhance(10.0)


enhanced_im.save('resized.jpg')
img_array = np.asarray(enhanced_im)
img_array = img_array / 255



input_shape = (256,256,3)



# reshape for model
# original model was trained with (32,32,3) 

img_array = img_array.reshape((1,256,256,3))

# modelo = Sequential()
# modelo.add(Conv2D(32, (3, 3), activation=activator, padding='same', input_shape=input_shape))
# modelo.add(Conv2D(32, (3, 3), activation=activator, padding='same'))
# modelo.add(Conv2D(32, (3, 3), activation=activator, padding='same'))
# modelo.add(MaxPooling2D((3, 3)))
# modelo.add(Dropout(0.2))
# modelo.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
# modelo.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
# modelo.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
# modelo.add(MaxPooling2D((3, 3)))
# modelo.add(Dropout(0.2))
# modelo.add(Conv2D(128, (3, 3), activation=activator, padding='same'))
# modelo.add(Conv2D(128, (3, 3), activation=activator, padding='same'))
# modelo.add(MaxPooling2D((3, 3)))
# modelo.add(Flatten())
# modelo.add(Dense(128, activation=activator))
# modelo.add(Dropout(0.2))
# modelo.add(Dense(10, activation='softmax'))

# modelo.compile(loss='categorical_crossentropy',optimizer=optimizer)
model = tf.keras.models.load_model('dnn/model_tl.h5')
model.load_weights('dnn/test2_tl.h5')
model.compile(loss='categorical_crossentropy',optimizer=optimizer)


# validate the 'activator'
pass
# validate the 'optimizer'
pass

# Load weights based on activator and optimizer
# probably not needed as we are already passing the optimizer as a variable 

# if optimizer == 'adam':
#     # compile with adam 
#     modelo.compile(loss='categorical_crossentropy',optimizer=optimizer)

#     # activator 
#     if activator == 'relu':
#         # load adam-relu
#         modelo.load_weights('dnn/relu-adam2.hdf5')
#     elif activator == 'sigmoid':
#         # load sigmoid-adam
#         modelo.load_weights('dnn/sigmoid-adam2.hdf5')
#     elif activator == 'tanh':
#         # load tanh-adam
#         modelo.load_weights('dnn/tanh-adam2.hdf5')
#     else:
#         print('error')
# elif optimizer == 'sgd':
#     # compile with sgd
#     modelo.compile(loss='categorical_crossentropy',optimizer=optimizer)
#     if activator == 'relu':
#         # load relu-sgd
#         modelo.load_weights('dnn/relu-sgd2.hdf5')
#     elif activator == 'sigmoid':
#         # load sigmoid-sgd
#         modelo.load_weights('dnn/sigmoid-sgd2.hdf5')
#     elif activator == 'tanh':
#         # load tanh-sgd
#         modelo.load_weights('dnn/tanh-sgd2.hdf5')

# else: 
#     print('error')

# Get the classification




############# classification ##############
# pred = modelo.predict(img_array)
# pred = pred[0]
# pred_class = class_labels[np.argmax(pred)]
pred = model.predict(img_array)
pred = pred[0]
pred_class = class_labels[np.argmax(pred)]

print(pred_class)

############# JSON ###############
# classification = {k:v for k,v in zip(class_labels,pred)}
classification = [
    {
        class_labels[0] : pred[0]
    },
    {
        class_labels[1] : pred[1]
    },
    {
        class_labels[2] : pred[2]
    },
    {
        class_labels[3] : pred[3]
    },
    {
        class_labels[4] : pred[4]
    }
]



########## output ################
print(classification)


