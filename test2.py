# first block runs
from PIL import Image,ImageFile
from matplotlib.pyplot import imshow
import requests
import numpy as np
from skimage.transform import resize


img = Image.open('datcat.jpg')
img_array = np.asarray(img)


# second block resize
img_array = resize(img_array,(32,32,3))
# img_array.shape

# third block
img_array = img_array.reshape((1,img_array.shape[0],img_array.shape[1],img_array.shape[2]))



# fourth block 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,Activation,MaxPooling2D

# fifth block
num_classes = 10
class_labels = ['airplane','automobile','bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# sixth block 
activator = 'relu'
optimizer = 'adam'
modelo_ar = Sequential()
modelo_ar.add(Conv2D(32, (3, 3), activation=activator, padding='same', input_shape=(32, 32, 3)))
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


# seventh block...
modelo_ar.compile(loss='categorical_crossentropy',optimizer='adam')
modelo_ar.load_weights('project3relu-adam2.hdf5')

# ninth block 
pred = modelo_ar.predict(img_array)
pred = np.argmax(pred)
pred = class_labels[pred]
print('Prediction:{}'.format(pred))













