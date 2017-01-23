from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import math
import numpy as np
from PIL import Image         
import cv2                 
import matplotlib.pyplot as plt
                                                      

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
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*45),int(h/2)),(0,255,0),thickness=4)
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
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are 
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    new_img = img[35:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    # normalize
    new_img = (new_img - 128.) / 128.
    return new_img

def random_distort(img):
    # random brightness
    new_img = img.astype(np.float)
    value = np.random.uniform(-0.5, 0.5)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 1.0 
    if value < 0:
        mask = (new_img[:,:,0] + value) < -1.0
    new_img[:,:,0] += np.where(mask, 0.0, value)
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/6,h/6)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return new_img


# Gather data
for row in driving_data:
    # skip it if ~0 speed or < 0.01 steering angle magnitude - not representative of driving behavior
    if float(row[6]) < 0.1 or abs(float(row[3])) < 0.01:
        continue
    # get, process, append center image
    img = cv2.imread(row[0])
    img = preprocess_image(img)
    img = random_distort(img)
    X.append(img)
    y.append(float(row[3]))
    # flip horizontally and add inverse steer angle
    flipped_img = cv2.flip(img, 1)
    X.append(flipped_img)
    y.append(-1.0*float(row[3]))
    # get, process, append left image
    imgL = cv2.imread(row[1])
    imgL = preprocess_image(imgL)
    imgL = random_distort(imgL)
    X.append(imgL)
    y.append(float(row[3])+0.2)
    # flip horizontally and add inverse steer angle
    flipped_imgL = cv2.flip(imgL, 1)
    X.append(flipped_imgL)
    y.append(-1.0*(float(row[3])+0.2))
    # get, process, append right image
    imgR = cv2.imread(row[2])
    imgR = preprocess_image(imgR)
    imgR = random_distort(imgR)
    X.append(imgR)
    y.append(float(row[3])-0.2)
    # flip horizontally and add inverse steer angle
    flipped_imgR = cv2.flip(imgR, 1)
    X.append(flipped_imgR)
    y.append(-1.0*(float(row[3])-0.2))

X = np.array(X)
y = np.array(y) 

print(X.shape, y.shape)

# display the dataset - a sort of sanity check
#dataset_to_video(X,y)

# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 23
avg_samples_per_bin = len(y)/num_bins
hist, bins = np.histogram(y, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(y), np.max(y)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
keep_probs = []
for i in range(num_bins):
    if hist[i] < avg_samples_per_bin:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/avg_samples_per_bin))
# going to have to remove starting from end
remove_list = []
for i in range(len(X)):
    for j in range(num_bins):
        if y[i] > bins[j] and y[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
X = np.delete(X, remove_list, axis=0)
y = np.delete(y, remove_list)

# print a histogram to see which steering angle ranges are most overrepresented
hist, bins = np.histogram(y, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(y), np.max(y)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

print(X.shape, y.shape)

###### ConvNet Definintion ######

# for debugging purposes - don't want to mess with the model if just checkin' the data
just_checkin_the_data = False

if not just_checkin_the_data:
    model = Sequential()

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', input_shape=(66, 200, 3)))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))

    # Add a dropout layer
    model.add(Dropout(0.50))

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and one dropout)
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.50))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(10, activation='tanh'))

    # Add a fully connected output layer
    model.add(Dense(1))

    # Add a ReLU activation layer
    #model.add(Activation('relu'))

    # Compile and train the model, 
    model.compile('adam', 'mean_squared_error', ['accuracy'])
    history = model.fit(X, y, batch_size=128, nb_epoch=5, validation_split=0.2, verbose=2)
    print(model.summary())

    # Save model data
    model.save_weights("./model.h5")
    json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)