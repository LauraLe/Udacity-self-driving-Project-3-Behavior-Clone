import cv2
import csv
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
 
def getDirsFromDrivingLogs(dataPath):
    """
    Returns the lines from a driving log 
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
    return lines


def getImagePaths(dataPath):
    """
    Finds all the images paths and steeing angles
    """
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    
    centerPaths = []
    leftPaths = []
    rightPaths = []
    angleList = []
    for directory in dataDirectories:
        lines = getDirsFromDrivingLogs(directory)
        center = []
        left = []
        right = []
        angle = []
        for line in lines:
            angle.append(float(line[3]))
            center.append( line[0].strip())
            left.append( line[1].strip())
            right.append(line[2].strip())
        centerPaths.extend(center)
        leftPaths.extend(left)
        rightPaths.extend(right)
        angleList.extend(angle)

    return (centerPaths, leftPaths, rightPaths, angleList)

def getAllImages(centerPaths, leftPaths, rightPaths, angleList, correction):
    """
    Combine the images from `center`, `left` and `right` using the correction factor 
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(centerPaths)
    imagePaths.extend(leftPaths)
    imagePaths.extend(rightPaths)
    steers = []
    steers.extend(angleList)
    steers.extend([x + correction for x in steers]) # correction for left image
    steers.extend([x - correction for x in steers]) # correction for right image
    return (imagePaths, steers)

import sklearn

def generator(samples, batch_size=32):
    """
    Generate the images and steer angles for training
    """
    num_samples = len(samples)
    
    while True: 
        samples = sklearn.utils.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for path, angle in batch_samples:
                inputImage = cv2.imread(path)
                image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)
                
                # flipping images
                images.append(cv2.flip(image,1))
                angles.append(-1.0 * angle )

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)


def preprocessLayers():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #normalize
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def nvidiaModel():
    """
    Creates nvidia autonomous car model
    """
    model = preprocessLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu',border_mode='valid'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu',border_mode='valid'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu',border_mode='valid'))
    model.add(Convolution2D(64,3,3, activation='relu',border_mode='valid'))
    model.add(Convolution2D(64,3,3, activation='relu',border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1)) #logit output - steering angle
    return model

if __name__ == '__main__':
    # Reading images locations.
    centerPaths, leftPaths, rightPaths, angleList = getImagePaths('data')
    imagePaths, angles = getAllImages(centerPaths, leftPaths, rightPaths, angleList, 0.2)
    
    # Splitting samples and creating generators.
   
    samples = list(zip(imagePaths, angles))
    train, validation = train_test_split(samples, test_size=0.2)
    
    print('Train samples: {}'.format(len(train)))
    print('Validation samples: {}'.format(len(validation)))
    
    train_generator = generator(train, batch_size=32)
    validation_generator = generator(validation, batch_size=32)
    
    model = nvidiaModel()
    
    # Train the model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch= \
                     len(train), validation_data=validation_generator, \
                     nb_val_samples=len(validation), nb_epoch=5, verbose=1)
    
    model.save('model.h5')
  
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

