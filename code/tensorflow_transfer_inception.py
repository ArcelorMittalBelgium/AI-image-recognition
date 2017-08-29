__author__ = 'flvdcast'

import os
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage import transform
import tensorflow as tf

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
slim = tf.contrib.slim

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
images, images_validation, labels, y_validation = train_test_split(X, y, test_size=0.33, random_state=0)

images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
images28 = rgb2gray(images28)

###############################
#
# Training information
#
###############################
num_epochs = 1

batch_size = 8  # if you get out of memory you can decrease the batch_size

initial_learning_rate = 0.0002  # you can play with these parameters
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

log_dir = './log'

checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'

image_size = 299


def run():
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    #======================= TRAINING PROCESS =========================
    #Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        train_data_directory = "train"

        images_orig, labels,num_samples, num_classes = load_data(train_data_directory)
        images = [transform.resize(image, (28, 28)) for image in images_orig]
        images = np.array(images)
        images = rgb2gray(images)

        #Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        #Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):   # here you can choose another network to start from
            logits, end_points = inception_resnet_v2(images, num_classes = num_classes, is_training = True)

        #Define the scopes that you want to exclude for restoration
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, num_classes)

        #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)


        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)


        #Run the managed session
        with sv.managed_session() as sess:
            for step in xrange(num_steps_per_epoch * num_epochs):
                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    print 'logits: \n', logits_value
                    print 'Probabilities: \n', probabilities_value
                    print 'predictions: \n', predictions_value
                    print 'Labels:\n:', labels_value

                #Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                #If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            #We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    run()