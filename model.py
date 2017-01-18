from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import math
import numpy as np
from PIL import Image         
import cv2                                                                       

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

import csv
# Import driving data from csv
with open('./training_data/driving_log.csv', newline='') as f:
    driving_data = list(csv.reader(f))

X = []
y = []

def displayCV2(img):
    '''
    Utility method to display a CV2 Image
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# example of opening/displaying image w/ pillow
#img = Image.open('test.png')
#img.show() 

# example of opening/displaying image w/ cv2
#img = cv2.imread('test.png')
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def preprocess_image(img):
    '''
    Method for preprocessing images:
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    new_img = img[35:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    # normalize
    new_img = (new_img - 128.) / 128.
    return new_img

for row in driving_data[0:10]:
    img = cv2.imread(row[0])
    img = preprocess_image(img)
    X.append(img)
    y.append(row[3])

X = np.array(X)
y = np.array(y)    

print(X.shape, y.shape)

model = Sequential()

# Add a convolution layer
model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(66, 200, 3)))

# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))

# Add a dropout layer
model.add(Dropout(0.50))

# Add a ReLU activation layer
model.add(Activation('relu'))

# Add a convolution layer
model.add(Convolution2D(36, 5, 5, border_mode='valid'))

# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))

# Add a dropout layer
model.add(Dropout(0.50))

# Add a ReLU activation layer
model.add(Activation('relu'))

# Add a convolution layer
model.add(Convolution2D(400, 1, 1, border_mode='valid'))

# Add a dropout layer
model.add(Dropout(0.50))

# Add a ReLU activation layer
model.add(Activation('relu'))

# Add a flatten layer
model.add(Flatten())

# Add a fully connected layer
model.add(Dense(128))

# Add a ReLU activation layer
model.add(Activation('relu'))

# Add a fully connected layer
model.add(Dense(84))

# Add a ReLU activation layer
model.add(Activation('relu'))

# Add a fully connected layer
model.add(Dense(1))

# Compile and train the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=12, validation_split=0.2, verbose=2)

# Save model data
model.save_weights("./model.h5")
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)