import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

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



def optimizer(learning_rate):
    """
    Use an Adam optimizer to reduce the cost with a given learning rate.

    Args:
    loss: output of loss function, the cross entropy loss
    learning_rate: how fast we want to change our weights

    Return: 
    tensorflow operation to optimize loss
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    return optimizer 



def calc_gradient(optimizer, loss):
    """
    Use an Adam optimizer to calculate the gradient with a given learning rate.
    We calculate and apply the gradients separately because we also want to 
    find the norm of the gradient to use as a stopping criteria if it falls 
    below a certain threshold.

    Args:
    optimizer
    loss

    Return:
    gradients and variables in a list of tuples
    """

    grads_and_vars = optimizer.compute_gradients(loss)
    return grads_and_vars



def gradient_norm(grads_and_vars):
    """
    Compute the gradient norm. On Euclidean space R^n, we use the L2 
    (Euclidean) norm, which measures the distance from the origin of 
    point x.

    Args:
    grad_and_vars: gradients calculated by the optimizer. Gradient and Variable
    is a list of tuples (gradient, and variable). We are only interested in the
    gradient.

    Returns:
    The norm of the gradient
    """

    # Calculate the norm for each training example
    gradient_norms = [tf.nn.l2_loss(g) for g, v in grads_and_vars]

    # Sum up the norms 
    gradient_norm = tf.add_n(gradient_norms)
    return gradient_norm 



def apply_gradient(grads_and_vars, optimizer):

    training_step = optimizer.apply_gradients(grads_and_vars)
    return training_step



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