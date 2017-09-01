---
title: "Tensorflow Basics"
sequence: 2
---

Originating from Google Brain Team, Tensorflow is now an open source library
for numerical computation using data flow graphs. By defining computations
in a graph, it avoids the overhead associated with transfering data between
Python and the specialised numerical native libraries. This is a similar
approach as used in Theano or Torch.


Lets take a look at the most important concepts.

References:
- [API docs](https://www.tensorflow.org/api_docs/python/)
- [Tensorflow getting started](https://www.tensorflow.org/get_started/)

# Import

```python
import tensorflow as tf
```

# Variables, placeholders, constants and operations

Since you are building a calculation graph, everything you define are in fact references to values that will be filled in once you run your calculation.

Constants are values that will not be changed.

```python
# Constant 1-D Tensor populated with value list.
# [1 2 3 4 5 6 7]
bias = tf.constant([1, 2, 3, 4, 5, 6, 7])

# Constant 2-D tensor populated with scalar value -1.
# [[-1. -1. -1.]
#  [-1. -1. -1.]]
bias = tf.constant(-1.0, shape=[2, 3])
```
 
Variables are values that will be changed by training algorithms.

```python
# Create a variable.
# [1, 2, 3]
weights = tf.Variable([1, 2, 3])

# Create a variable with random initiated values.
# Different on every run, eg: [[ 0.04587448  0.06301729]]
weights = tf.Variable(tf.truncated_normal([1,2], stddev=0.1))
```

Placeholders are values whose values will be provided when starting a calculation.

```python
# The second argument is the shape of the input, it is used
# to check dimensions in advance. Use None to not specify a dimension.
input = tf.placeholder(tf.float32, [None, 28*28])
```

An operation is a calculation graph node that is the result
of performing a calculation. It takes zero or more tensors as
input and outputs zero or more tensors.

```python
# Use the variable in the graph like any Tensor.
y = tf.matmul(input, weights) + bias
```

# (Interactive) session and the computational graph

A session allows you to evaluate the nodes in the computational graph.
A session holds resources that should be released when it is not longer needed.

```python
# Define operations in advance
tensor = tf.Variable(tf.truncated_normal([2, 2]))

# Run the graph
# Using the `close()` method.
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # Initialize variables before evaluation.
print(sess.run(tensor.eval())
sess.close()

# Using "with" sets the "default sessions", making things a bit shorter
with tf.Session() as sess:
    tf.global_variables_initializer().run() # Initialize variables before evaluation.
    print(tensor.eval())
```

TODO: Filling in placeholders

For working in a shell or notebook, you can use the `InteractiveSession`, which sets itself as default session on construction.

```python
# Define operations in advance
tensor = tf.Variable(tf.truncated_normal([2, 2]))

# Create a session
sess = tf.InteractiveSession()

# Crunch numbers
tf.global_variables_initializer().run() # Initialize variables before evaluation.
print(tensor.eval())
```

# Saving progress

You can save variables during the calculation.

```python
weights = tf.Variable(initial_value=[0])
learning_delta = tf.constant(value=[1])

# Mocked training algorithm: adds delta to weights on every execuction
train_op = tf.assign_add(weights, learning_delta)

saver = tf.train.Saver([weights])
save_path = os.path.join(os.curdir, "save-folder", "save-filename")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # Learning loop
    for i in range(1000):
        train_op.eval()
        if (i % 100 == 0):
            print("Weight is %s" % weights.eval())
            saver.save(sess, save_path, global_step=i)
```

Output:

```
Weight is [1]
Weight is [101]
Weight is [201]
Weight is [301]
Weight is [401]
Weight is [501]
Weight is [601]
Weight is [701]
Weight is [801]
Weight is [901]
```

This will have created a folder `save-folder` containing snapshots of the `weights` variable, which can be used to restore the variable values at a later point.

```python
with tf.Session() as sess:
    # print(weights.eval()) # Would fail, since this is a new session
    tf.global_variables_initializer().run() # Not needed for restoring
    print(weights.eval())
    saver.restore(sess, './save-folder/save-filename-500')
    print(weights.eval())
```

Output:

```
[0]
INFO:tensorflow:Restoring parameters from ./save-folder/save-filename-500
[501]
```

# GPU assignment


# Tensorboard
TensorBoard is a suite of web applications for inspecting and understanding your TensorFlow runs and graphs. It helps engineers to analyze, visualize, and debug TensorFlow graphs. The advantage of this add-on is that you don't have to write your own visualization tools for the loss curve or the training and validation curves. 

First you have to define the log directory, there you will store the log files
```python
writer = tf.train.SummaryWriter(log_path, graph=tf.get_default_graph())
```

The following code will write the cost and the accuracy for each batch run
```python
# create a summary for our cost and accuracy
tf.scalar_summary("cost", cross_entropy)
tf.scalar_summary("accuracy", accuracy)

# merge the different summaries to one 
summary_op = tf.merge_all_summaries()

# perform the operations we defined earlier on batch
_, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})
            
# write log data to the summary 
writer.add_summary(summary, epoch * batch_count + i)
```

Secondly you type in the terminal the following command and the Tensorboard get started on your localhost (http://localhost:6006).
```
tensorboard --logdir ='/home/logpath'
```

Finally, you have some nice visualizations of your learning curve or accuracy and the other variables that you have defined.

[https://www.tensorflow.org/get_started/summaries_and_tensorboard]()
[http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/]()

# Examples

- MNIST (using linear regression): [Tutorial](https://www.tensorflow.org/get_started/mnist/beginners) [code](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
- MNIST (using a deep net): [Tutorial](https://www.tensorflow.org/get_started/mnist/pros) [code](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py)