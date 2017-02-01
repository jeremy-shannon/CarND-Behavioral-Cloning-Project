# Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project

*My solution to the Udacity Self-Driving Car Engineer Nanodegree behavioral cloning project.*

This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo). 

## Introduction

The object of this project is to apply deep learning principles to a simulated driving application. The simulator includes both training and autonomous modes. In training mode, user generated driving data is collected in the form of simulated car dashboard camera images and conrol data (steering angle, throttle, brake, speed). Using the Keras deep learning framework, a model is produced using the collected driving data (see model.py) and saved as model.json (with neural network weights saved as model.h5). 

Using the saved model, drive.py (provided by Udacity, but amended to finetune conrols) starts up a local server to control the simulator in autonomous mode. The command to run the server is `python drive.py model.json`

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
