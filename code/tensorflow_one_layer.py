__author__ = 'flvdcast'

import os
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage import transform
import tensorflow as tf
import skimage.data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"]="1" #work only on one GPU

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    label_tag = 0
    label_dict ={}

    num_samples = 0
    num_classes = 0

    for d in directories:

        sub_directories = [k for k in os.listdir(os.path.join(data_directory, d))]

        for s in sub_directories:
            label_directory = os.path.join(data_directory,d,s)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if f.endswith('.png')]

            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(label_tag)
                num_samples += 1
        label_tag += 1

    label_dict[d] = label_tag
    return images, labels, num_samples, label_tag

train_data_directory = 'train'

X, y, testt,num_classes = load_data(train_data_directory)
# do cross validation; create training and validation data, test data is not labelled and cannot been used
images_train, images_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=0)
################################
#
# Load the trainingsdata an make uniform
#
################################

# Rescale the images in the `images` array to equal size
images28 = [transform.resize(image, (28, 28)) for image in images_train]

# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)

####################
#
# Add data augmentation
#
####################

# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, num_classes, tf.nn.relu) # only one FCN layer

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

################################
#
# Train the model
#
################################

tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: y_train})
        if i % 10 == 0:
            print("Loss: ", loss)

################################
#
# Test the accuracy of the model
#
################################

#load the validation data

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in images_validation]

# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(y_validation, predicted)])

# Calculate the accuracy
accuracy = match_count / len(y_validation)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))