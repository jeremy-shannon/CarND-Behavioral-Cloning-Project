from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import numpy as np
from PIL import Image         
import cv2                 
import matplotlib.pyplot as plt
from os import getcwd
import csv
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# example of opening/displaying image w/ pillow
#img = Image.open('test.png')
#img.show() 

# example of opening/displaying image w/ cv2
#img = cv2.imread('test.png')
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def displayCV2(img):
    '''
    Utility method to display a CV2 Image
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_img_for_visualization(image, angle, pred_angle, frame):
    '''
    Used by visualize_dataset method to format image prior to displaying. Converts colorspace back to original BGR, applies text to display steering angle and frame number (within batch to be visualized), and applies lines representing steering angle and model-predicted steering angle (if available) to image.
    '''    
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)
    # apply a line representing the steering angle
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0),thickness=4)
    if pred_angle is not None:
        cv2.line(img,(int(w/2),int(h)),(int(w/2+pred_angle*w/4),int(h/2)),(0,0,255),thickness=4)
    return img
    
def visualize_dataset(X,y,y_pred=None):
    '''
    format the data from the dataset (image, steering angle) and display
    '''
    for i in range(len(X)):
        if y_pred is not None:
            img = process_img_for_visualization(X[i], y[i], y_pred[i], i)
        else: 
            img = process_img_for_visualization(X[i], y[i], None, i)
        displayCV2(img)        

def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are 
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    #new_img = img[35:140,:,:]
    # crop to 40x320x3
    new_img = img[50:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # scale to ?x?x3
    #new_img = cv2.resize(new_img,(80, 10), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def random_distort(img, angle):
    ''' 
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)

def generate_training_data(image_paths, angles, batch_size=128, validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images, then yield them to the
    model. if 'validation_flag' is true the image is not distorted. also flips images with turning angle magnitudes of greater than 0.33, as to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])
    while True:       
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = preprocess_image(img)
            if not validation_flag:
                img, angle = random_distort(img, angle)
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)
            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([],[])
                    image_paths, angles = shuffle(image_paths, angles)


def generate_training_data_for_visualization(image_paths, angles, batch_size=20, validation_flag=False):
    '''
    method for loading, processing, and distorting images
    if 'validation_flag' is true the image is not distorted
    '''
    X = []
    y = []
    image_paths, angles = shuffle(image_paths, angles)
    for i in range(batch_size):
        img = cv2.imread(image_paths[i])
        angle = angles[i]
        img = preprocess_image(img)
        if not validation_flag:
            img, angle = random_distort(img, angle)
        X.append(img)
        y.append(angle)
    return (np.array(X), np.array(y))

'''
Main program 
'''

# select data source(s) here
using_my_data = True
using_udacity_data = True

data_to_use = [using_my_data, using_udacity_data]
img_path_prepend = ['', getcwd() + '/udacity_data/']
csv_path = ['./training_data/driving_log.csv', './udacity_data/driving_log.csv']

image_paths = []
angles = []

for j in range(2):
    if not data_to_use[j]:
        # 0 = my own data, 1 = Udacity supplied data
        print('not using dataset ', j)
        continue
    # Import driving data from csv
    with open(csv_path[j], newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))


    # Gather data - image paths and angles for center, left, right cameras in each row
    for row in driving_data[1:]:
        # skip it if ~0 speed - not representative of driving behavior
        if float(row[6]) < 0.1 :
            continue
        # get center image path and angle
        image_paths.append(img_path_prepend[j] + row[0])
        angles.append(float(row[3]))
        # get left image path and angle
        image_paths.append(img_path_prepend[j] + row[1])
        angles.append(float(row[3])+0.25)
        # get left image path and angle
        image_paths.append(img_path_prepend[j] + row[2])
        angles.append(float(row[3])-0.25)

image_paths = np.array(image_paths)
angles = np.array(angles)

print('Before:', image_paths.shape, angles.shape)

# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 23
avg_samples_per_bin = len(angles)/num_bins
hist, bins = np.histogram(angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#plt.show()

# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(angles, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#plt.show()

print('After:', image_paths.shape, angles.shape)

# visualize a single batch of the data
X,y = generate_training_data_for_visualization(image_paths, angles)
visualize_dataset(X,y)

# split into train/test sets
image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
                                                                                  test_size=0.05, random_state=42)
print('Train:', image_paths_train.shape, angles_train.shape)
print('Test:', image_paths_test.shape, angles_test.shape)

###### ConvNet Definintion ######

# for debugging purposes - don't want to mess with the model if just checkin' the data
just_checkin_the_data = False

if not just_checkin_the_data:
    model = Sequential()

    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    #model.add(Dropout(0.50))
    
    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))

    # Compile and train the model, 
    #model.compile('adam', 'mean_squared_error')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    ############  just for tweaking model ##############
    # pulling out 128 random samples and training just on them, to make sure the model is capable of overfitting
    # indices_train = np.random.randint(0, len(image_paths_train), 128)
    # indices_test = np.random.randint(0, len(image_paths_test), 12)
    # image_paths_train = image_paths_train[indices_train]
    # angles_train = angles_train[indices_train]
    # image_paths_test = image_paths_test[indices_test]
    # angles_test = angles_test[indices_test]
    #############################################################

    # initialize generators
    train_gen = generate_training_data(image_paths_train, angles_train, validation_flag=False, batch_size=64)
    val_gen = generate_training_data(image_paths_train, angles_train, validation_flag=True, batch_size=64)
    test_gen = generate_training_data(image_paths_test, angles_test, validation_flag=True, batch_size=64)

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    #history = model.fit(X, y, batch_size=128, nb_epoch=5, validation_split=0.2, verbose=2)
    history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=2560, samples_per_epoch=23040, 
                                  nb_epoch=5, verbose=2, callbacks=[checkpoint])
    # print('Test Loss:', model.evaluate_generator(test_gen, 128))

    print(model.summary())

    # visualize some predictions
    n = 12
    X_test,y_test = generate_training_data_for_visualization(image_paths_test[:n], angles_test[:n], batch_size=n,                                                                     validation_flag=True)
    y_pred = model.predict(X_test, n, verbose=2)
    visualize_dataset(X_test, y_test, y_pred)

    # Save model data
    model.save_weights('./model.h5')
    json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)