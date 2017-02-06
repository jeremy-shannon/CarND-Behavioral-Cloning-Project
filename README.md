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

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (and later in the student forum, the [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py)) - the diagram below is a depiction of the nVidia model.

<img src="./images/nVidia_model.png?raw=true" width="400px">

First I reproduced this model as depicted in the image - including image normalization using a Keras Lambda function, with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text - including converting from RGB to YUV color space, and 2x2 striding on the 5x5 convolutional layers. The paper does not mention any sort of activation function or means of mitigating overfitting, so I began with `tanh` activation functions on each fully-connected layer, and dropout (with a keep probability of 0.5) between the two sets of convolution layers and after the first fully-connected layer. 

### 2. Collecting Additional Driving Data

Udacity provides a dataset that can be used alone to produce a working model. However, students are encouraged (and let's admit, it's more fun) to collect our own. Particulaly, Udacity encourages including "recovery" data while training. This means that data should be captured starting from the point of approaching the edge of the track (perhaps nearly missing a turn and almost driving off the track) and recording the recovery process to give the model a chance to learn recovery behavior. It's easy enough for experienced humans to drive the car reliably around the track, but if the model has never experienced being too close to the edge and then finds itself in just that situation it won't know how to react.

### 3. Loading and Preprocessing

In training mode, the simulator produces three images per frame while recording corresponding to left-, right-, and center-mounted cameras, each giving a different perspective of the track ahead. The simulator also produces a `csv` file which includes file paths for each of these images, along with steering angle, throttle, brake, and speed for each frame. My algorithm loads the file paths for all three camera views for each frame, along with the angle (adjusted by +0.25 for the left frame and -0.25 for the right), into two numpy arrays `image_paths` and `angles`.

Images produced by the simulator in training mode are 320x160, and therefore require preprocessing prior to being fed to the CNN because it expects input images to be size 200x66. To achieve this, the bottom 20 pixels and the top 35 pixels (although this number later changed) are cropped from the image and it is then resized to 200x66. A subtle Gaussian blur is also applied and the color space is converted from RGB to YUV. Because `drive.py` uses the same CNN model to predict steering angles in real time, it requires the same image preprocessing (**Note, however: using `cv2.imread`, as `model.py` does, reads images in BGR, while images received by `drive.py` from the simulator are RGB, and thus require different color space conversion**). All of this is accomplished by methods called `preprocess_image` in both `model.py` and `drive.py`.

### 4. Jitter

To minimize the model's tendency to overfit to the conditions of the test track, images are "jittered" before being fed to the CNN. The jittering (implemented using the method `random_distort`) consists of a randomized brightness adjustment, a randomized shadow, and a randomized horizon shift. The shadow effect is simply a darkening of a random rectangular portion of the image, starting at either the left or right edge and spanning the height of the image. The horizon shift applies a perspective transform beginning at the horizon line (at roughly 2/5 of the height) and shifting it up or down randomly by up to 1/8 of the image height. The horizon shift is meant to mimic the hilly conditions of the challenge track. The effects of the jitter can be observed in the sample to the right.

<img src="./images/sanity-check-take-4.gif?raw=true" style="float: right; padding: 20px;">

### 5. Data Visualization

An important step in producing data for the model, espeically when preprocessing (and even more so when applying any sort of augmentation or jitter) the data, is to visualize it. This acts as a sort of sanity check to verify that the preprocessing is not fundamentally flawed. Flawed data will almost certainly act to confuse the model and result in unacceptable performance. For this reason, I included a method 'visualize_dataset', which accepts a numpy array of images `X`, a numpy array of floats `y` (steering angle labels), and an optional numpy array of floats `y_pred` (steering angle predictions from the model). This method calls `process_img_for_visualization` for each image and label in the arrays.

The `process_img_for_visualization` method accepts an image input, float `angle`, float `pred_angle`, and integer `frame`, and it returns an annotated image ready for display. It is used by the `visualize_dataset` method to format an image prior to displaying. It converts the image colorspace from YUV back to the original BGR, applies text the the image representing the steering angle and frame number (within the batch to be visualized), and applies lines representing the steering angle and the model-predicted steering angle (if available) to the image.

<img src="./images/data_distribution_before_3.png?raw=true" style="float: right; padding: 20px; max-width: 400px">

### 6. Data Distribution Flattening 

Because the test track includes long sections with very slight or no curvature, the data captured from it tends to be heavily skewed toward low and zero turning angles. This creates a problem for the neural network, which then becomes biased toward driving in a straight line and can become easily confused by sharp turns. The distribution of the input data can be observed to the right, the black line represents what would be a uniform distribution of the data points.

To reduce the occurrence of low and zero angle data points, I first chose a number of bins (I decided upon 23) and produced a histogram of the turning angles using `numpy.histogram`. I also computed the average number of samples per bin (`avg_samples_per_bin` - what would be a uniform distribution) and plotted them together. Next, I determined a "keep probability" (`keep_prob`) for the samples belonging to each bin. That keep probability is 1.0 for bins that contain less than `avg_samples_per_bin`, and for other bins the keep probability is calculated to be the number of samples for that bin divided by `avg_samples_per_bin` (for example, if a bin contains twice the average number of data points its keep probability will be 0.5). Finally, I removed random data points from the data set with a frequency of `(1 - keep_prob)`. 

<img src="./images/data_distribution_after.png?raw=true" style="float: right; padding: 20px; max-width: 400px">

The resulting data distribution can be seen in the chart to the right. The distribution is not uniform overall, but it is much closer to uniform for lower and zero turning angles.

*After implementing the above strategies, the resulting model performed very well - driving reliably around the test track multiple times. It also navigated the challenge track quite well, until it encountered an especially sharp turn. The following strategies were adopted primarily to improve the model enough to drive the length of the challenge track, although not all of the them contributed to that goal directly.*

### 7. Implementing a Python Generator in Keras

When working with datasets that have a large memory footprint (large quantities of image data, in particular) Keras python generators are a convenient way to load the dataset one batch at a time rather than loading it all at once. Although this was not a problem for my implementation, because the project rubric made mention of it I felt compelled to give it a try. 

The generator `generate_training_data` accepts as parameters a numpy array of strings `image_paths`, a numpy array of floats `angles`, an integer `batch_size` (default of 128), and a boolean `validation_flag` (default of `False`). Loading the numpy arrays `image_paths` (string) and `angles` (float) from the csv file, as well as adjusting the data distribution (see "Data Distribution Flattening," above) and splitting the data into training and test sets, is still done in the main program. 

`generate_training_data` shuffles `image_paths` and `angles`, and for each pair it reads the image referred to by the path using `cv2.imread`. It then calls `preprocess_image` and `random_distort` (if `validation_flag` is `False`) to preprocess and jitter the image. If the magnitude of the steering angle is greater than 0.33, another image is produced which is the mirror image of the original using `cv2.flip` and the angle is inverted - this helps to reduce bias toward low and zero turning angles, as well as balance out the instance of higher angles in each direction so neither left nor right turning angles become overrepresented. Each of the produced images and corresponding angles is added to a list and when the lengths of the lists reach `batch_size` the lists are converted to numpy arrays and yielded to the calling generator from the model. Finally, the lists are reset to allow another batch to be built and `image_paths` and `angles` are again shuffled.

`generate_training_data` runs continuously, returning batches of image data to the model as it makes requests, but it's important to view the data that is being fed to the model, as mentioned above in "Data Visualization." That's the purpose of the method `generate_training_data_for_visualization`, which returns a smaller batch of data to the main program for display. (*This turned out to be critical, at one point revealing a bug in my implementation of `cv2.flip` causing the image to be flipped vertically instead of horizontally*)

### 8. More Aggressive Cropping

Inspired by [David Ventimiglia's post](http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html?fb_comment_id=1429370707086975_1432730663417646&comment_id=1432702413420471&reply_comment_id=1432730663417646#f2752653e047148) (particularly where he says "For instance, if you have a neural network with no memory or anticipatory functions, you might downplay the importance of features within your data that contain information about the future as opposed to features that contain information about the present."), I began exploring more aggressive cropping during the image preprocessing stage. This also required changes to the convolutional layers in the model, resulting in a considerably smaller model footprint with far fewer parameters. Unfortunately, I was not successful implementing this approach (although the reason may have been because of an error in the `drive.py` image preprocessing), and ultimately returned to the original nVidia model.

### 9. Cleaning the dataset

<img src="./images/sanity-check-take-5.gif?raw=true" style="float: right; padding: 20px;">

Another mostly unsuccessful attempt to improve the model's performace was inspired by [David Brailovsky's post](https://medium.freecodecamp.com/recognizing-traffic-lights-with-deep-learning-23dae23287cc#.linb6gh1d) describing his competition-winning model for identifying traffic signals. In it, he discovered that the model performed especially poorly on certain data points, and then found those data points to be mislabeled in several cases. I created `clean.py` which leverages parts of both `model.py` and `drive.py` to display frames from the dataset on which the model performs the worst. The intent was to manually adjust the steering angles for the mislabeled frames, but this approach was tedious, and often the problem was with the model's prediction and not the label or the ideal ground truth lay somewhere between the two.

### 10. Futher Model Adjustments

Some other strategies implemented to combat overfitting and otherwise attempt to get the car to drive more smoothly are (these were implemented mostly due to consensus from the nanodegree community, and not necessarily all at once):
- Removing dropout layers and adding L2 regularization (`lambda` of 0.001) to all model layers, convolutional and fully-connected
- Removing `tanh` activations on fully-connected layers and adding `ELU` activations to all model layers, convolutional and fully-connected
These strategies did, indeed, result in less bouncing back and forth between the sides of the road, particularly on the test track where the model was most likely to overfit to the recovery data.

### 11. Further Data Distribution Flattening

At one point, I had decided I might be throwing out too much of my data trying to achieve a more uniform distribution. So instead of discarding data points until the distribution for a bin reaches the would-be average for all bins, I made the target *twice* the would-be average for all bins. The resulting distribution can be seen in the chart below on the left. This resulted in a noticeable bias toward driving straight, particularly on the challenge track. 

The consensus from the nanodegree community was that underperforming on the challenge track most likely meant that there was not a high enough frequency of higher steering angle data points in the dataset. I once again adjusted the flattening algorithm, setting target maxiumum count for each bin to *half* of the would-be average for all bins. The histogram depicting the results of this adjustment can be seen in the chart below and to the right.

<img src="./images/data_distribution_after_3.png?raw=true" style="float: left; padding: 20,0; max-width: 50%">
<img src="./images/data_distribution_after_4.png?raw=true" style="float: right; padding: 20,0; max-width: 50%">
<div style="float: left;">

## Results 

These strategies resulted in a 