# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:10:34 2016

@author: Jeff
"""

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os

# Function to map pixel values to -1 to 1 range
def normalize_grayscale(image_data):
    return image_data / 127.5 - 1

# Declare arrays to temporarily store image and steering angle data  nnnv
img = []
label = []

# Open .csv file containing training features and labels
f = open('driving_log.csv')
csv_f = csv.reader(f)


# Read the .csv file line by line
# If the steering angle is 0, do nothing, otherwise:
# 1. Store the steering angle in the label array
# 2. Convert the color of the image and store in img
# 3. Flip the steering angle and image, and store each
# 4. Add L and R camera images with an offset steering anlge to represent additional data

for row in csv_f:
    angle = float(row[3])
    
    if angle == 0:
        ...
        # If steering angle = 0, do nothing
    else:
        # Add original and flipped steering angle
        label.append(angle)
        label.append(-1 * angle)
        
        
        # Add original and flipped image
        image = cv2.imread(row[0]).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = normalize_grayscale(image.astype(np.float32))
        image2 = np.fliplr(image)
        
        img.append(image)
        img.append(image2)
        
        # Add left camera angles along with a steering angle offset
        label.append(angle + 0.08)
        imageL = cv2.imread(row[1]).astype('uint8')
        imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
        imageL = normalize_grayscale(imageL.astype(np.float32))
        img.append(imageL)
        
        # Add right camera angles along with a steering angle offset
        label.append(angle - 0.08)
        imageR = cv2.imread(row[2]).astype('uint8')
        imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)
        imageR = normalize_grayscale(imageR.astype(np.float32))
        img.append(imageR)
    
# Convert training data to numpy arrays and rename
X_train = np.array(img, np.float32)
y_train = np.array(label, np.float32)
# Trim the training image to remove anything above the horizion
X_train = X_train[:,60:,:,:]

# Sanity Check
print('Training label count: ')
print(len(y_train))
print('Training feature count: ')
print('Length = ')
print(len(X_train))
plt.imshow(X_train[-1], 'gray')
plt.show()

# Store training data in a pickle file
pickle_file = 'training_sim_2016-12-03'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file')
    try:
        with open('training_sim_2016-12-03', 'wb') as pfile:
            pickle.dump(
                {
                     'train_dataset' : X_train,
                     'train_labels' : y_train,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
print('Data cached in pickle file')