# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:10:34 2016

@author: Jeff
"""

import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.optimizers import Adam
import json

# Load stored training data from pickle file
pickle_file = 'training_sim_2016-12-03'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    del pickle_data

# Calculate training feature dimensions    
img_shape = X_train[0].shape
print('Training data image size is: ')
print(img_shape)

# Split validation data from the training data
# 95% training data, 5% validation data
X_train, X_valid, y_train, y_valid = train_test_split(
                                                      X_train,
                                                      y_train,
                                                      test_size = 0.05,
                                                      random_state = 832289)


# Define network graph in Keras
# Referencing Comma.ai model
# 1. Convolution layer with 16 8x8 filters, 4x4 skim, and ReLU activation
# 2. Convolution layer with 32 5x5 filters, 2x2 skim, and ReLU activation
# 3. Convolution layer with 64 5x5 filters, 2x2 skim, and ReLU activation
# 4. Flatten the features
# 5. Dropout, 20%
# 6. Fully connected layer with 512 nodes and ReLU activation
# 7. Dropout, 50%
# 8. Fully connected layer with 1 node, output is steering angle
model = Sequential()


model.add(Convolution2D(16, 8, 8, subsample = (4, 4), input_shape = img_shape, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5, subsample = (2, 2), border_mode = 'same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 5, 5, subsample = (2, 2), border_mode = 'same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))

# Define Hyperparameters
batch_size = 16
nb_epoch = 4


# Compile model using MSE loss function and defaultAdam optimizer
model.summary()

model.compile(loss = 'mean_squared_error',
              optimizer = Adam())

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

# Sanity check
print(model.predict(X_train[550:560]))
print(y_train[550:560])

