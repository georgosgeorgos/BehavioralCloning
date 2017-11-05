# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
*  Use the simulator to collect data of good driving behavior
*  Build, a convolution neural network in Keras that predicts steering angles from images
*  Train and validate the model with a training and validation set
*  Test that the model successfully drives around track one without leaving the road
*  Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image" 

---
### Files

This project includes the following files:
* model.py containing the script to train the model
* utils.py containing a Python generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md

#### Run

Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. I used two model 

### Architectures

I used two model architecture for this assignment: the NVIDIA architecture (defined into NVIDIA_model.py) and a Squeeze model with simple bypass (defined into SqueezeNet.py).

The NVIDIA model is a classical architecture with convolutional layers and fully connected layers.
The Squeeze model is composed only by convolutional layers and fire modules. 

To reduce overfitting I applied dropout and data augmentation.
I also applied batch-normalization.
The model used an adam optimizer, so the learning rate was not tuned manually.

#### Training Data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road (in particular for the second track).

### Design

For the first track I decided to use the modified NVIDIA model.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 cropped and normalized  				| 	
| Convolution 5x5     	| 2x2 stride, valid padding 
| Convolution 5x5     	| 2x2 stride, valid padding 
| Convolution 5x5     	| 2x2 stride, valid padding 
| Convolution 5x5     	| 2x2 stride, valid padding 
| Convolution 5x5     	| 2x2 stride, valid padding 
| Flatten 				| 
| Dropout				| 0.5
| Fully connected		| hidden units 100
| Fully connected		| hidden units 50
| Fully connected		| hidden units 10
| Fully connected		| hidden units 1 			

 ...


#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track two using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.


After the collection process, I had ~ 12k data points. I then preprocessed this data by cropping the images and normalizing.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 


#### Reference


[NVIDIA model](https://arxiv.org/pdf/1604.07316.pdf)  
[SqueezeNet model](https://arxiv.org/pdf/1602.07360.pdf)

