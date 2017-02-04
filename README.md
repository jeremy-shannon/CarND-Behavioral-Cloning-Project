# Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project

*My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.*

**Note: This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo).**

## Introduction

The object of this project is to apply deep learning principles to a simulated driving application. The simulator includes both training and autonomous modes, and two tracks on which the car can be driven - I will refer to these as the "test track" (which is the track from which training data is collected and on which the output is evaluated for class credit) and the "challenge track" (which includes hills, tight turns, and other features not included in the test track). 

In training mode, user generated driving data is collected in the form of simulated car dashboard camera images and conrol data (steering angle, throttle, brake, speed). Using the Keras deep learning framework, a convolutional neural network (CNN) model is produced using the collected driving data (see `model.py`) and saved as `model.json` (with CNN weights saved as `model.h5`). 

Using the saved model, drive.py (provided by Udacity, but amended slightly to ensure compatiblity with the CNN model and to finetune conrols) starts up a local server to control the simulator in autonomous mode. The command to run the server is `python drive.py model.json`; the model weights are retrieved using the same name but the extension `.h5` (i.e. `model.h5`).

The challenge of this project is not only developing a CNN model that is able to drive the car around the test track without leaving the track boundary, but also feeding training data to the CNN in a way that allows the model to generalize well enough to drive in an environment it has not yet encountered (i.e. the challenge track). 

## Approach

### 1. Base Model and Adjustments

<img src="./images/nvidia_model.png?raw=true" style="float: right; padding: 20px; max-width: 400px">

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (and later in the student forum, the [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py)) - the diagram to the right is a depiction of the nVidia model.

First I reproduced this model as depicted in the image - with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text - including converting from RGB to YUV color space, and 2x2 striding on the 5x5 convolutional layers. The paper does not mention any sort of activation function or means of mitigating overfitting, so I began with `tanh` activation functions on each fully-connected layer, and dropout (with a keep probability of 0.5) between the two sets of convolution layers. 

### 2. Loading and Preprocessing

In training mode, the simulator produces three images per frame while recording corresponding to left-, right-, and center-mounted cameras, each giving a different perspective of the track ahead. The simulator also produces a `csv` file which includes file paths for each of these images, along with steering angle, throttle, brake, and speed for each frame. My algorithm loads the file paths for all three camera views for each frame, along with the angle (adjusted by +0.25 for the left frame and -0.25 for the right), into two numpy arrays `image_paths` and `angles`.

Images produced by the simulator in training mode are 320x160, and therefore require preprocessing prior to being fed to the CNN because it expects input images to be size 200x66. To achieve this, I cropped the bottom 20 pixels and the top 35 pixels (although this number later changed) from the image and then resized it to 200x66. I also applied a subtle Gaussian blur and converted from RGB to YUV color space. Because `drive.py` uses the same CNN model to predict steering angles in real time, it requires the same image preprocessing (**Note, however: using `cv2.imread`, as `model.py` does, reads images in BGR, while images received by `drive.py` from the simulator are RGB, and thus require different color space conversion**).

### 3. Jitter

<img src="./images/sanity-check-take-4.gif?raw=true" style="float: right; padding: 20px;">

To minimize the model's tendency to overfit to the conditions of the test track, images are "jittered" before being fed to the CNN. The jittering (implemented using the method `random_distort`) consists of a randomized brightness adjustment, a randomized shadow, and a randomized horizon shift. The shadow effect is simply a darkening of a random rectangular portion of the image, starting at either the left or right edge and spanning the height of the image. The horizon shift applies a perspective transform beginning at the horizon line (at roughly 2/5 of the height) and shifting it up or down randomly by up to 1/8 of the image height. The horizon shift is meant to mimic the hilly conditions of the challenge track. The effects of the jitter can be observed in the sample to the right.

### 4. Distribution Adjustment 

<img src="./images/data_distribution_before_3.png?raw=true" style="float: right; padding: 20px; max-width: 400px">

Because the test track includes long sections with very slight or no curvature, the data captured from it tends to be heavily skewed toward low and zero turning angles. This creates a problem for the neural network, which then becomes biased toward driving in a straight line and can become easily confused by sharp turns. The distribution of the input data can be observed to the right, the black line represents what would be a uniform distribution of the data points.

To reduce the occurrence of low and zero angle data points, I first chose a number of bins (I decided upon 23) and produced a histogram of the turning angles using `numpy.histogram`. I also computed the average number of samples per bin (`avg_samples_per_bin` - what would be a uniform distribution) and plotted them together. Next, I determined a "keep probability" (`keep_prob`) for the samples belonging to each bin. That keep probability is 1.0 for bins that contain less than `avg_samples_per_bin`, and for other bins the keep probability is calculated to be the number of samples for that bin divided by `avg_samples_per_bin` (for example, if a bin contains twice the average number of data points its keep probability will be 0.5). Finally, I removed random data points from the data set with a frequency of `(1 - keep_prob)`. 

<img src="./images/data_distribution_after_2.png?raw=true" style="float: right; padding: 20px; max-width: 400px">

The resulting data distribution can be seen in the chart to the right. The distribution is not uniform overall, but it is much closer to uniform for lower and zero turning angles.

*After implementing the above strategies, the resulting model performed very well - driving reliably around the test track multiple times. It also drove on the challenge track quite well, until it encountered an especially sharp turn. The following strategies were adopted primarily to improve the model enough to drive the length of the challenge track, although not all of the them contributed to that goal directly.*




- removing most ~0 angle data points to get a more uniform distribution of angles
- implementing generator
- moving normalization to within model
- more aggressive crop (no horizon) and shrink
 - remove horizon shifting and blurring
 - adjustments to model to accommodate smaller input size
- bugs caused by RGB/BGR
- less aggresive crop (still no horizon)
- clean.py to identify poorly labeled data points
- combine my data with udacity data
- including left/right cameras or not (smoother without)
- not removing as many ~0 angle data points
- adding L2 regularization to fully connected layers
- removing dropout (negligible)
- removing tanh activations on FC layers and adding ELU to all layers
- original crop with horizon shifting and blurring (bringin' it back!)
- removing more low steering angle data points for an even MORE uniform distribution
