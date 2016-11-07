from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from vcg import VCG

# Parameters
learning_rate = 0.1
training_iters = 1500
batch_size = 20
display_step = 100

# Network Parameters
num_input = 3
num_steps = 120
num_hidden = 7
num_classes = 5



def inference(sequence, weights, biases, num_hidden):
    """
    Forward propagation on LSTM neural network with linear activation.

    Args:
    sequence: sequence with (x, y, z) at each time step for 120 time steps.
    weights: dimensions (num_hidden x num_classes)
    biases: dimensions (num_classes x 1)

    Return:
    Output of linear classification, dimension (batch_size x num_classes)
    """

    # Define a lstm cell with tensorflow
    cell = rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.dynamic_rnn(cell, sequence, dtype=tf.float32)

    # Permuting batch_size and num_steps
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Linear activation on last output, using rnn inner loop last output
    linear_activation = tf.matmul(outputs[-1], weights) + biases
    return linear_activation



def loss(logit, target):
    """
    Find the average loss in a batch with cross entropy. First applies
    softmax on logits (unscaled values) to create a probability distribution,
    then use cross entropy to find the loss.

    Args:
    logit: unscaled values, output of inference
    target: correct probability distribution (one hot vector)

    Return:
    tensorflow operation for average loss, what we are optimizing to reduce.
    """

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, target))
    return cost 



def optimize(loss, learning_rate):
    """
    Use an Adam optimizer to reduce the cost with a given learning rate.

    Args:
    loss: output of loss function, the cross entropy loss
    learning_rate: how fast we want to change our weights

    Return: 
    tensorflow operation to optimize loss
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer 



def accuracy(logit, target):
    """
    Determine the percentage accuracy of the network.

    Args:
    logit: unscaled values that the model predicted, output of inference
    target: correct probability distribution, one hot vector

    Return:
    accuracy as a percentage
    """

    # Grab the highest value from each row and compare to target
    correct = tf.equal(tf.argmax(logit, 1), tf.argmax(target, 1))

    # Cast boolean values to float and average
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # return average accuracy
    return accuracy 

# Initialize Placeholders and Variables
# ===========================================================================

# Initialize dataset iterator
data_sets = VCG("sequence.npy", "target.npy", 0.6, 0.2, 0.2, True)

# Declare placeholders for VCG and corresponding labels
sequence = tf.placeholder(
                dtype=tf.float32, 
                shape=[None, num_steps, num_input],
                name="vcg_sequence"
            )
target = tf.placeholder(
                dtype=tf.float32, 
                shape=[None, num_classes],
                name="dyssynchrony_index"
            )

# Declare weights used for linear activation on output of network
weights = tf.Variable(tf.random_normal([num_hidden, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))



# Define TensorFlow Operations
# ===========================================================================

# Assigns values for each example across all classes corresponding to 
# likelihood that the sequence belongs to that class (logits are unscaled
# values, not a probability distribution)
logit = inference(sequence, weights, biases, num_hidden)

# Initialize tensorflow operations to train network
loss = loss(logit, target)                      # Loss function
optimizer = optimize(loss, learning_rate)       # Update weights
accuracy = accuracy(logit, target)              # Determine accuracy

# Initializing the variables
init = tf.initialize_all_variables()



# Run TensorFlow Session
# ===========================================================================
# Launch the session to perform ONE training step as a test
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Grab the first batch
    batch_x, batch_y = data_sets.train.next_batch(batch_size)

    # Evaluate loss before any training
    print("Loss (before training): %4f." % sess.run(
                    loss, 
                    feed_dict={sequence: batch_x, target: batch_y}))

    # Determine accuracy
    print("Training Accuracy: %2f" % sess.run(
                    accuracy, 
                    feed_dict={sequence: batch_x, target: batch_y}))

    # Run one training step
    sess.run(optimizer, feed_dict={sequence: batch_x, target: batch_y})

    # Evaluate loss after one training step
    print("Loss (after training): %4f." % sess.run(
                    loss, 
                    feed_dict={sequence: batch_x, target: batch_y}))

    # Determine accuracy
    print("Training Accuracy: %2f" % sess.run(
                    accuracy, 
                    feed_dict={sequence: batch_x, target: batch_y}))

