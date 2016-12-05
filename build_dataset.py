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

#X_train = np.array([], np.uint8)
#y_train = np.array([], np.uint8)
img = []
label = []

f = open('driving_log.csv')

csv_f = csv.reader(f)
i = 0
for row in csv_f:
    angle = float(row[3])
    
    if angle == 0:
        ...
        #if i >1:
        #    label.append(angle)
        #    
        #    image = cv2.imread(row[0]).astype('uint8')
        #    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #    img.append(image)
            
        #    i = 0
        #else:
        #    i += 1
    else:
        label.append(angle)
        label.append(-1 * angle)

        image = cv2.imread(row[0]).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2 = cv2.flip(image,1)
        
        img.append(image)
        img.append(image2)
    #y_train = np.append(y_train, row[3])
    #if len(y_train) < 1000:
    #print(len(y_train))

    #image2 = image
        #plt.imshow(image, 'gray')
        #plt.show()
        #print(image.shape)
        #X_train = np.append(X_train, [image])
        #print(img[0][0][0])
    #if len(X_train) < 5:

#for row in csv_f:
#    label.append(float(row[3]))
#    image = cv2.imread(row[0]).astype('uint8')
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    img.append(image)

    #print(row[0])
X_train = np.array(img, np.uint8)
y_train = np.array(label, np.float32)

def normalize_grayscale(image_data):
    return image_data / 127.5 - 1

print(X_train[0])
print(X_train[0].min())
print(np.mean(X_train[0]))
print(X_train[0] / 127.5 - 1)
print(X_train[0].shape)
    
X_train = normalize_grayscale(X_train.astype(np.float32))

#np.fromstring(y_train, dtype = 'uint8')    
print(y_train[0])
print(y_train[-25:-15])
print(len(y_train))
#print(type(y_train[0]))

print('Length = ')
print(len(X_train))
plt.imshow(X_train[-1], 'gray')
plt.show()

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