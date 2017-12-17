# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/loss_3epochs.png "Loss after 3 epochs"
[image3]: ./images/loss_5epochs.png "Loss after 5 epochs"
[image4]: ./images/driving_central.png "Driving at Center Image"
[image5]: ./images/driving_reverse.png "Driving Reverse Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md or writeup_report.pdf summarizing the results
* images: visualization of some recorded data and model
* drive_track1: video clip driving in autonomous of track 1 is uploaded on youtube link [here](https://youtu.be/fArwvZ_fpmI)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used model of [NIVIDA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
My model summary is below. 
It consists of 1 lambda layer to normalize and cropping the inputs (crop 50 from the top and 20 from the bottom), 5 convolution layers and 3 fully connected layer.

<prep>
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
</prep>
Total params: 981,819
Trainable params: 981,819

The model includes RELU layers to introduce nonlinearity in 5 convolution layers, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

I don't use dropout. I train the model using 5 epochs
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:
* Udacity provided data
* My own data including : 3 laps of centered driving, 2 laps of reverse driving (to prevent the model bias to left steering), 3 laps driving across the bridge
I try to include data driving in recovered mode i.e when car drive to left or right, recover it back to center but that make my car in autonomous mode drive in zigzag. Exclude data driving in recovered mode from training sample helps the model's performance
Without 4-5 laps driving across the bridge, the car will hit and stuck to the left once reach the end of the bridge 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use NVIDIA model on inputs without cropping and normalization steps. The training takes extreme long time. Cropping helps to reduce training time significantly but car would run out of track. Addining normalization helps. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by ratio 80%-20%. I first train for 3 epochs and the car would run very badly on the bridge and stuck to the end of the bridge. The training loss and validation loss looks as in 

![Image][image2]

It look like there is an overfitting as the validation loss is higher than the training loss

To combat the overfitting, I added more training data, recording only when car go through the bridges. I then train in 5 epochs. After that, the car run well on track 1 and as the result, I did not consider drop out.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The loss after 5 epochs look as below

![Image][image3]

#### 2. Final Model Architecture

The final model architecture visualization is below

![Image][image1]

#### 3. Creation of the Training Set & Training Process

Data samples has
Total Images: 65664
Train samples: 52531
Validation samples: 13133

To capture good driving behavior, I first recorded two-three laps on track one using center lane driving. Here is an example image of center lane driving:

![Image][image4]
The 1st row are original images of center, left, right camera. The 2nd row are corresponding images after preprocess by normalize and crop 50 from the top & 20 from bottom

![Image][image5]
Above image show of images of reverse driving

Then I repeated driving at center on track two in order to get more data points.

To augment the data sat, I also flipped images and reverse angles thinking that this would help with the left-turn bias

 I also use center, left, right images. I adjust the steering angle of left & right image based on the centered image steering angle as following
 * left's angle = center's angle + 0.2
 * right's angle = center's angle - 0.2

After the collection process, I had 65664 number of data points. I then preprocessed this data by normalize and crop to get only image part where the road is

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the decrease in MSE on validation set. Ideally I can get the ideal number of epochs by setting callback in keras and stop once the MSE on validation set does not improved after certain number of epochs. For the simplicity, I use 5 epochs and the model works. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The model.h5 perform well on Track 1 but did not work on Track 2.

I cannot use python drive.py model.h5 run1 to save image to a folder because that worsen my model performance. Hence I use QuickPlayer to record my screen
