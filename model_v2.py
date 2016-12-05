# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:10:34 2016

@author: Jeff
"""

#import csv
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
import json

# Load stored training data from pickle file
pickle_file = 'training_sim_2016-12-03'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    del pickle_data
    
img_shape = X_train[0].shape
print(img_shape)

# Split validation data from the training data
X_train, X_valid, y_train, y_valid = train_test_split(
                                                      X_train,
                                                      y_train,
                                                      test_size = 0.05,
                                                      random_state = 832289)



model = Sequential()
model.add(Convolution2D(24, 5, 5,
                        border_mode = 'valid',
                        subsample = (2, 2),
                        input_shape = img_shape))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.75))
model.add(Convolution2D(36, 5, 5,
                        subsample = (2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.75))
model.add(Convolution2D(48, 5, 5,
                        subsample = (2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.75))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.75))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.75))
model.add(Dense(50))
model.add(Activation('relu'))
#model.add(Dropout(0.75))
model.add(Dense(10))
model.add(Activation('relu'))
#model.add(Dropout(0.75))
model.add(Dense(1))

batch_size = 16
nb_epoch = 1

model.summary()

model.compile(loss = 'mean_squared_error',
              optimizer = Adam(),
              metrics = ['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size = batch_size,
                    nb_epoch = nb_epoch,
                    verbose = 1,
                    validation_data = (X_valid, y_valid))


# serialize model to JSON
model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print(model.predict(X_train[1000:1005]))
print(y_train[1000:1005])

