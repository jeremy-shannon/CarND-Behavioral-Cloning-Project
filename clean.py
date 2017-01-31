import argparse
import base64
import json
import cv2

import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

def displayCV2(img):
    '''
    Utility method to display a CV2 Image
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_img_for_video(image, angle, pred_angle, frame):
    '''
    Used by visualize_dataset method to format image prior to adding to video
    '''    
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=.5, color=(255,255,255), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(255,255,255), thickness=1)
    # apply a line representing the steering angle
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0),thickness=4)
    if pred_angle is not None:
        cv2.line(img,(int(w/2),int(h)),(int(w/2+pred_angle*w/4),int(h/2)),(0,0,255),thickness=4)
    return img
    
def visualize_dataset(X,y,y_pred=None):
    '''
    format the data from the dataset (image, steering angle) and place it into a video file 
    '''
    for i in range(len(X)):
        if y_pred is not None:
            img = process_img_for_video(X[i], y[i], y_pred[i], i)
        else: 
            img = process_img_for_video(X[i], y[i], None, i)
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
    new_img = img[80:140,:,:]
    # apply subtle blur
    #new_img = cv2.GaussianBlur(new_img, (5,5), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # scale to ?x?x3
    #new_img = cv2.resize(new_img,(80, 10), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def generate_training_data_for_visualization(image_paths, angles, batch_size=20, validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images
    if 'validation_flag' is true the image is not distorted
    '''
    X = []
    y = []
    for i in range(batch_size):
        img = cv2.imread(image_paths[i])
        angle = angles[i]
        img = preprocess_image(img)
        if not validation_flag:
            img, angle = random_distort(img, angle)
        X.append(img)
        y.append(angle)
    return (np.array(X), np.array(y))


if __name__ == '__main__':
    '''
    This little guy mostly takes bits from drive.py and model.py to help clean up some data, pulling the data points
    that generate the most erroneous predictions from the model and visualizing them (to make sure they're actually bad)
    so I can then edit the actual steering angle values in the csv file 
    '''
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    using_udacity_data = False
    img_path_prepend = ''
    csv_path = './training_data/driving_log.csv'
    if using_udacity_data:
        img_path_prepend = getcwd() + '/udacity_data/'
        csv_path = './udacity_data/driving_log.csv'

    import csv
    # Import driving data from csv
    with open(csv_path, newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    image_paths = []
    angles = []

    # Gather data - image paths and angles for center, left, right cameras in each row
    for row in driving_data[1:]:
        # skip it if ~0 speed - not representative of driving behavior
        if float(row[6]) < 0.1 :
            continue
        # get center image path and angle
        image_paths.append(img_path_prepend + row[0])
        angles.append(float(row[3]))

    image_paths = np.array(image_paths)
    angles = np.array(angles)

    print('shapes:', image_paths.shape, angles.shape)

    # visualize some predictions
    n = 12
    X_test,y_test = generate_training_data_for_visualization(image_paths[:n], angles[:n], batch_size=n,                                                                     validation_flag=True)
    y_pred = model.predict(X_test, n, verbose=2)
    #visualize_dataset(X_test, y_test, y_pred)

    # get predictions on a larger batch - basically pull out worst predictions from each batch so they can be 
    # corrected manually in the csv
    n = 1000
    for i in reversed(range(len(image_paths)//n + 1)):
        start_i = i * n
        end_i = (i+1) * n
        batch_size = n
        if end_i > len(image_paths):
            end_i = len(image_paths)
            batch_size = end_i - start_i - 1
        X_test,y_test = generate_training_data_for_visualization(image_paths[start_i:end_i], 
                                                                 angles[start_i:end_i], 
                                                                 batch_size=batch_size,                                                                     
                                                                 validation_flag=True)
        y_pred = model.predict(X_test, n, verbose=2).reshape(-1,)
        # sort the diffs between predicted and actual, then take the bottom m indices
        m = 5
        bottom_m = np.argsort(abs(y_pred-y_test))[batch_size-m:]
        print('indices:', bottom_m+(i*n) + 1)
        print('actuals:', y_test[bottom_m])
        print('predictions:', y_pred[bottom_m])
        print('')
        visualize_dataset(X_test[bottom_m], y_test[bottom_m], y_pred[bottom_m])
                                                            