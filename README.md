# Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project

My solution to the Udacity Self-Driving Car Engineer Nanodegree behavioral cloning project.

This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo). 

The object of this project is to apply deep learning principles to a simulated driving application. The simulator includes both training and autonomous modes. In training mode, driving data is collected in the form of images and conrol data (steering angle, throttle, brake, speed). Using the Keras deep learning framework, a model is produced using the collected driving data (see model.py) and saved as model.json (with neural network weights saved as model.h5). 

Using the saved model, drive.py (provided by Udacity, but amended to finetune conrols) starts up a local server to control the simulator in autonomous mode. 