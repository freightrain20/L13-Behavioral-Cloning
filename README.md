# L13-Behavioral-Cloning
##Building the training data (build_dataset.py)

**Collecting Data**

My simulation data consisted of:
  - (3) laps driven around the track correctly
  - (2) laps correcting from the right side of the track to the center
  - (2) laps correcting from the left side of the track to the center

**Training Data Preprocessing**

I chose to do a few things to preprocess the simulator data before building the training dataset:
  - First, I chose to use features/labels if the steering angle was nonzero. This prevented the model from overfitting to a zero steering angle at all times.
  - Second, I needed to train for right turns and increase the robustness of the dataset. I did this by:
    - adding a flipped image along with -1 * steering angle 
    - adding the left camera image along with 0.08 + steering angle
    - adding the right camera image along with 0.08 - steering angle
  - Third, I limited the graph's tendancy to train to items in the image backgroun by removing the top 60 rows from the image. As a result the graph will not be biased by background scenery.
  - The data is then stored in a pickle file for used in model.py

##Building the graph (model.py)

**Loading the data**

I load the data from the pickle file and then split it into training and validation sets. I use a 95% training and 5% validation split as it appears to show enough variation for the validation set to be meaningful.

**Defining the graph**

After experimenting with transfer learning and my own designs, I settled on a modified version of Comma.ai's graph found at (https://github.com/commaai/research/blob/master/train_steering_model.py). The final architecture is:

  1. Convolution layer with 16 8x8 filters, 4x4 skim, and ReLU activation
  2. Convolution layer with 32 5x5 filters, 2x2 skim, and ReLU activation
  3. Convolution layer with 64 5x5 filters, 2x2 skim, and ReLU activation
  4. Flatten the features
  5. Dropout, 20%
  6. Fully connected layer with 512 nodes and ReLU activation
  7. Dropout, 50%
  8. Fully connected layer with 1 node, output is steering angle
  
  - I found that ReLU activation provided the smoothest result when driving in the simulation.
  - Batch normalization tended to prevent my model from converging without overfitting

**Defining the Hyperparameters**
  - Batch size: 16 - I found that smaller batch sizes tended to produce better results on the track
  - Epochs: 4 - Increasing the epochs tended to cause the model to settle at a very small steering angle regardless of the image. This seemed to be the sweet spot of providing enough information for the model to train without finding a single solution that minimized the cost function for all images.
  - Cost function: Mean Squared Error - Since this is a regression problem, I needed a cost function that would provide a continuous error value. MSE seemed to work well.
  - Optimizer: Adam with default parameters - I found the default learning rate, 4 epochs, and 16 batch size to work best on my computer

**Training the Graph**

The MSE for training and validation data could provide a general indication of performance, but is not indicative of true performance. I noticed that the lowest MSE value tended to be when the model had determined a near zero steering angle for all features, which would not work well in the simulation. In order to get around this, I would include a 10 item sanity check which compared expected and predicted steering angles for a given feature. I did not expect the steering angles to match, but I wanted to see that the predicted steering angles were changing and ballpark correct. If the steering angles looked reasonable, I would try the model in the simulation

##Simulating the Graph (drive.py)
I needed to do some quick preprocessing on the images from the simulation in order to match the training set:
  1. Map image pixel values to the range of -1 to 1
  2. Remove the top 60 rows from the image

##Summary

Overall, I am happy with the results. I tried to keep my project simple by limiting the complexity of the graph and keeping preprocessing to a minimum when running the model on the simulator. Instead, I relied on building quality training data to build an accurate model. My model was able to navigate the Left road course for hours but struggled with the Right dirt course. I think the model would be better equipped to handle the dirt course if I expanded the training dataset by creating new images of varying hues and brightness levels from my original training set.
