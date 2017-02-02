# Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project

*My solution to the Udacity Self-Driving Car Engineer Nanodegree behavioral cloning project.*

**Note: This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo).**

## Introduction

The object of this project is to apply deep learning principles to a simulated driving application. The simulator includes both training and autonomous modes, and two tracks on which the car can be driven - I will refer to these as the "test track" (which is the track from which training data is collected and on which the output is evaluated for class credit) and the "challenge track" (which includes hills, tight turns, and other features not included in the test track). 

In training mode, user generated driving data is collected in the form of simulated car dashboard camera images and conrol data (steering angle, throttle, brake, speed). Using the Keras deep learning framework, a convolutional neural network (CNN) model is produced using the collected driving data (see `model.py`) and saved as `model.json` (with CNN weights saved as `model.h5`). 

Using the saved model, drive.py (provided by Udacity, but amended slightly to ensure compatiblity with the CNN model and to finetune conrols) starts up a local server to control the simulator in autonomous mode. The command to run the server is `python drive.py model.json`; the model weights are retrieved using the same name but the extension `.h5` (i.e. `model.h5`).

The challenge of this project is not only developing a CNN model that is able to drive the car around the test track without leaving the track boundary, but also feeding training data to the CNN in a way that allows the model to generalize well enough to drive in an environment it has not yet encountered (i.e. the challenge track). 

## Approach

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (and later in the student forum, the [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py))

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
