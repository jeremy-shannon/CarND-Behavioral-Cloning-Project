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
    driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

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

def process_img_for_video(image, angle, frame):
    '''
    Used by dataset_to_video method to format image prior to adding to video
    '''    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # undo nomalization
    img = (128*image+128).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=.5, color=(255,255,255), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(255,255,255), thickness=1)
    # apply a line representing the steering angle
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*25),int(h/2)),(0,255,0),thickness=4)
    return img
    
def dataset_to_video(X,y):
    '''
    format the data from the dataset (image, steering angle) and place it into a video file 
    '''
    # there are 6 versions of the same scene: 3 cameras, and flipped versions of each
    for j in range(6):
        for i in range(j,len(X),6):
            img = process_img_for_video(X[i], y[i], i)
            displayCV2(img)        

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
    # random brightness
    new_img = new_img.astype(np.float)
    new_img[:,:,0] *= np.random.uniform(0.75, 255/np.max(new_img[:,:,0]))
    # normalize
    new_img = (new_img - 128.) / 128.
    return new_img

# Gather data
for row in driving_data[0:20]:
    # get, process, append center image
    img = cv2.imread(row[0])
    img = preprocess_image(img)
    X.append(img)
    y.append(float(row[3]))
    # flip horizontally and add inverse steer angle
    flipped_img = cv2.flip(img, 1)
    X.append(flipped_img)
    y.append(-1.0*float(row[3]))
    # get, process, append left image
    imgL = cv2.imread(row[1])
    imgL = preprocess_image(imgL)
    X.append(imgL)
    y.append(float(row[3])+0.15)
    # flip horizontally and add inverse steer angle
    flipped_imgL = cv2.flip(imgL, 1)
    X.append(flipped_imgL)
    y.append(-1.0*(float(row[3])+0.15))
    # get, process, append right image
    imgR = cv2.imread(row[2])
    imgR = preprocess_image(imgR)
    X.append(imgR)
    y.append(float(row[3])-0.15)
    # flip horizontally and add inverse steer angle
    flipped_imgR = cv2.flip(imgR, 1)
    X.append(flipped_imgR)
    y.append(-1.0*(float(row[3])-0.15))


X = np.array(X)
y = np.array(y) 

dataset_to_video(X,y)

print(np.histogram(y, 3))
print(np.histogram(y, 5))
print(np.histogram(y, 7))
print(np.histogram(y, 9))

print(X.shape, y.shape)

###### ConvNet Definintion ######

model = Sequential()

# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', input_shape=(66, 200, 3)))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))

# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))

# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10), tanh activation
model.add(Dense(100, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(10, activation='tanh'))

# Add a fully connected output layer
model.add(Dense(1))

# Add a dropout layer
#model.add(Dropout(0.50))
# Add a ReLU activation layer
#model.add(Activation('relu'))

# Compile and train the model, 
model.compile('adam', 'mean_squared_error', ['accuracy'])
#history = model.fit(X, y, batch_size=128, nb_epoch=5, validation_split=0.2, verbose=2)

# Save model data
model.save_weights("./model.h5")
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)