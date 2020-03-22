
# image processing
from PIL import Image,ImageFile
from matplotlib.pyplot import imshow
import requests
import numpy as np
from skimage.transform import resize

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,Activation,MaxPooling2D


# prediction as predecined by the data the model was trained on 
num_classes = 10
class_labels = ['airplane','automobile','bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


img = Image.open('datcat.jpg')
img_array = np.asarray(img)

input_shape = img_array.shape
img_array = resize(img_array,input_shape,anti_aliasing=True)
img_array = img_array.reshape((1,img_array.shape[0],img_array.shape[1],img_array.shape[2]))

# the model is called modelo_ar

activator = 'relu'
optimizer = 'adam'


modelo_ar = Sequential()
modelo_ar.add(Conv2D(32, (3, 3), activation=activator, padding='same', input_shape=input_shape))
modelo_ar.add(Conv2D(32, (3, 3), activation=activator, padding='same'))
modelo_ar.add(Conv2D(32, (3, 3), activation=activator, padding='same'))
modelo_ar.add(MaxPooling2D((2, 2)))
modelo_ar.add(Dropout(0.2))
modelo_ar.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
modelo_ar.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
modelo_ar.add(Conv2D(64, (3, 3), activation=activator, padding='same'))
modelo_ar.add(MaxPooling2D((2, 2)))
modelo_ar.add(Dropout(0.2))
modelo_ar.add(Conv2D(128, (3, 3), activation=activator, padding='same'))
modelo_ar.add(Conv2D(128, (3, 3), activation=activator, padding='same'))
modelo_ar.add(MaxPooling2D((2, 2)))
modelo_ar.add(Flatten())
modelo_ar.add(Dense(128, activation=activator))
modelo_ar.add(Dropout(0.2))
modelo_ar.add(Dense(10, activation='softmax'))
modelo_ar.compile(loss='categorical_crossentropy',optimizer=optimizer)

modelo_ar.compile(loss='categorical_crossentropy',optimizer='adam')
modelo_ar.load_weights('relu-adam.hdf5')


# prediction
# pred = modelo_ar.predict(img_array)
# pred = np.argmax(pred)
# pred = class_labels[pred]
# print('Prediction:{}'.format(pred))

print(img_array.shape)