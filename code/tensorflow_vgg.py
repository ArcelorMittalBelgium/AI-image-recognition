__author__ = 'flvdcast'

# bases on the code of omondrot https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
# added model saving
# added tensorboard visualization

import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import skimage.data
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='/media/florian/ArcelorMittal/vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=8, type=int)    # lower in case of memory problems
parser.add_argument('--num_workers', default=8, type=int)   # set equally to the amount of cpu processors
parser.add_argument('--num_epochs1', default=100, type=int)
parser.add_argument('--num_epochs2', default=100, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--saved_model', default='/media/florian/ArcelorMittal/vgg_finetuned_model', type=str)
parser.add_argument('--log_path', default='/media/florian/ArcelorMittal/logs', type=str)
VGG_MEAN = [123.68, 116.78, 103.94] # average pixel values

os.environ["CUDA_VISIBLE_DEVICES"]="3" #define a specific GPU that you want to work on

'''
Function to get the data paths and the annotated labels.
'''
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    label_tag = 0
    num_samples = 0

    for d in directories:

        sub_directories = [k for k in os.listdir(os.path.join(data_directory, d))]

        for s in sub_directories:
            label_directory = os.path.join(data_directory,d,s)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if f.endswith('.png')]

            for f in file_names:
                images.append(f)
                labels.append(label_tag)
                num_samples += 1
        label_tag += 1

    return images, labels


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    # Method to get the training or validation (is_training=False), accuracy
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})   #get the predicitons
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break
    acc = float(num_correct) / num_samples
    return acc

def main(args):
    # get the labelled trainings data paths
    train_data_directory = 'train'
    X, y = load_data(train_data_directory)

    # randomly split the data into a training and validation set (for cross validation purposes)
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(X, y, test_size=0.10, random_state=0)

    num_classes = len(set(train_labels))

    # create the Tensorflow graph
    graph = tf.Graph()
    with graph.as_default():

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)   # read the image input files and convert them to jpeg
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            # resale the images to be equally sized
            scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width,lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])
            return resized_image, label


        # VGG preprocessing steps for training (cropping, random flipping, subtracting the mean pixel values)
        def training_preprocess(image, label):
            crop_image = tf.random_crop(image, [224, 224, 3])
            flip_image = tf.image.random_flip_left_right(crop_image)

            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = flip_image - means

            return centered_image, label

        # VGG preprocessing steps for validation (cropping, subtracting the mean pixel values)
        def val_preprocess(image, label):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means

            return centered_image, label


        # Training dataset preprocessing and Tensorflow tensors adaptation
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_parse_function, num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(training_preprocess, num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        batched_train_dataset = train_dataset.batch(args.batch_size)

        # validation dataset preprocessing and Tensorflow tensors adaptation
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function, num_threads=args.num_workers, output_buffer_size=args.batch_size)
        val_dataset = val_dataset.map(val_preprocess,num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)

        # loop over al the images to get them properly preprocessed
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        is_training = tf.placeholder(tf.bool)   # define training or testing mode

        # get the vgg-16 model via the slim framework
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training, dropout_keep_prob=args.dropout_keep_prob)

        # load the pretrained model weights off vgg
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        with tf.name_scope('Model'):
                # get the pretrained weights for all layers except those from the last layer
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        with tf.name_scope('FC8_new'):

            # randomly initialize the weights of the final layer of the vgg-model
            fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
            fc8_init = tf.variables_initializer(fc8_variables)

        with tf.name_scope('loss'):
                # define the loss function
                tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                loss = tf.losses.get_total_loss()

        with tf.name_scope('SGD_fc8'):
            # set the learning rate and the optimization model for the final layer
            fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
            fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

        with tf.name_scope('SGD_all'):
            # set the learning rate for the general model (very small value as we are not retraining the mode from scratch)
            full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
            full_train_op = full_optimizer.minimize(loss)

        with tf.name_scope('Accuracy'):
            # Define the accuracy calculation of the trained model
            prediction = tf.to_int32(tf.argmax(logits, 1))
            correct_prediction = tf.equal(prediction, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

         # Create a summary to monitor loss tensor
        tf.summary.scalar("loss", loss)

        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", accuracy)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        tf.get_default_graph().finalize()

    config = tf.ConfigProto(log_device_placement=True)


    with tf.Session(config=config, graph=graph) as sess:

        # create a log file to store all the data and summaries
        summary_writer  = tf.summary.FileWriter(args.log_path, sess.graph)

        init_fn(sess)  # load the pretrained weights
        sess.run(fc8_init)  # Train the last new layer of the network

        for epoch in range(args.num_epochs1):   # run a couple of iterations
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            sess.run(train_init_op)
            while True:
                try:
                    _, summary  = sess.run([fc8_train_op,merged_summary_op], {is_training: True})
                    summary_writer.add_summary(summary, epoch)

                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)

            # Here you can add some code to do early stopping: validation accuracy is decreasing, while the trainings accuracy is increasing

        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(args.num_epochs2):
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            sess.run(train_init_op)
            while True:
                try:
                    _, summary = sess.run([full_train_op,merged_summary_op], {is_training: True})
                    summary_writer.add_summary(summary, epoch + args.num_epochs1)

                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)

            # Here you can add some code to do early stopping: validation accuracy is decreasing, while the trainings accuracy is increasing

            # Finally it is important to save the trained model

        #Create a saver object which will save the weights and variables of the trained model
        saver = tf.train.Saver()
        saver.save(sess, args.save_path, global_step=args.num_epochs2)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)