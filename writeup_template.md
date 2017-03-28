**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image_left]: ./examples/left_turn.jpg "left turn"
[image_right]: ./examples/right_turn.jpg "right turn"
[histogram]: ./examples/histogram.jpg "histogram"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model contains two preprocessing steps and a stack of convolution layer followed by fully connected layer. 

In the preprocessing steps, I normalize the the input between (-1,1) and crop 70 pixels from the top border and 25 pixels from the bottom.  

The neural network consists of a convolution neural network with 5x5 and 3x3 filter sizes. The depths are between 24 and 64. All the conv layers contain RELUs. After the conv layers, 4 fully connected layers are used. The dimension of the outputs are  100->50->10->1. To avoid overfitting, I add Dropout layer with a dropout rate of 0,5 between the FC layers.   

#### 2. Attempts to reduce overfitting in the model
* The model contains dropout layers in order to reduce overfitting.
* 6 recorded datasets are used to train the model. Two of them which represent two difficult turns are used to fine-tune the model. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model is first pre-trained using an adam optimizer with batch size 128, and refined with additional dataset with batch size 32 and a small learning rate 0.001->0.0005->0.0002

#### 4. Appropriate training data

I keep the training data in different directoies and use the panda library to select only the training data, the speed of which is in a specific range. For the first dataset, the speed range is [20,30], which most represent driving straightly. Other datasets contain more turns. I use the (speed>3) as the speed range for training.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional layers followed by fully connected layers

My first step was to use a convolution neural network model that I used in the traffic sign classification project. But it turns out it does not work very well and the parameters are too large which results in a quite longer training time. Then, I tried the model which is similar to the Nvidia model reported in the paper. And the size of the model is only about 4 MB.

Only using the center image for training will not work, because the car has to learn about how to recover drifting from the center. I also use the left and right camera images as the training data and generate an offset steering angle for them. The strategy that I use for generating the steering angle is based on the current steering angle of the center camera. I use a quadratic function to compute the offset, the more curvy a turn is, the more offset I give to each turn. The intuition behinds that is if the car drifts in the turn, it requires more steering than in the straight line.  

After training with the initial data, the car is able to drive almost all the situation including the bridge. However, it still can not pass a left turn and a right turn. These two turns have a sharp steering angle, and the dataset is under represented in the training set. Then, I record additional datasets which only contains these two turns, while lowing the learning rate at the same time. After about 100 epochs, the model converges.  Magically, the car is able to pass both turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

![alt text][image_left]

![alt text][image_right]

#### 2. Final Model Architecture

The final neural network consists of a convolution neural network with 5x5 and 3x3 filter sizes. The depths are between 24 and 64. All the conv layers contain RELUs. After the conv layers, 4 fully connected layers are used. The dimension of the outputs are  100->50->10->1. To avoid overfitting, I add Dropout layer with a dropout rate of 0,5 between the FC layers.   

#### 3. Creation of the Training Set & Training Process

I record the data using a ps3 joystick which allows me to generate smooth data especially for the turns. I record two complete laps with different driving directions so that the training data also contains right turns. I also record additional  data in which I most concentrate on the turns. During recording, I run the simulator at a lower speed, so I can fine control the car and get good training examples at the turn. The following histogram shows the distribution of steering angles for each training set.

![alt text][histogram]


I also tried using the flipped images to augment the data, however, in my case, it turns out help that much. The model performs worse than when I do not use these data. 

![alt text][image6]
![alt text][image7]

After the collection process, I had about 10000 number of data points. By using the left and right images, I have 3 times more training data. I randomly shuffled the data set and put 20% of the data into a validation set. 

