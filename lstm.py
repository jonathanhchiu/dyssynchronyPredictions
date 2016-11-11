from __future__ import print_function
import tensorflow as tf 
import numpy as np 
import math

from vcg import VCG
import ops 


# Network Hyperparameters and Dimensions
# ===========================================================================

# Parameters
learning_rate = 0.1
training_iters = 50
batch_size = 32
gradient_norm_threshold = 0.001

# Network Parameters
num_input = 3
num_steps = 120
num_hidden = 10
num_classes = 5

logs_path='data/'


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
weights = tf.Variable(
            tf.truncated_normal(
                shape=[num_hidden, num_classes],
                stddev= 1.0 / math.sqrt(float(num_classes))
            )
        )
biases = tf.Variable(
            tf.zeros([num_classes])
        )



# Define TensorFlow Operations
# ===========================================================================

# Assigns values for each example across all classes corresponding to 
# likelihood that the sequence belongs to that class (logits are unscaled
# values, not a probability distribution)
logit = ops.inference(sequence, weights, biases, num_hidden)

# Initialize tensorflow operations to train network:
# Loss function: cross entropy
loss = ops.loss(logit, target)

# Add a scalar summary for the snapshot loss.
tf.scalar_summary(loss.op.name, loss)

# Gradient calculations:
# Initialize gradient descent optimizer
optimizer = ops.optimizer(learning_rate)

# Step 1: Calculate gradient
gradients = ops.calc_gradient(optimizer, loss)

# Step 2: Calculate gradient norm for stopping criteria
gradient_norm = ops.gradient_norm(gradients)

# Step 3: Apply gradients and update model
training_step = ops.apply_gradient(gradients, optimizer)

# Post training: Determine accuracy
accuracy = ops.accuracy(logit, target)



# Setup Tensorboard
# ===========================================================================
writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

# Keep track of loss
tf.scalar_summary("Loss", loss)
summary_op = tf.merge_all_summaries()


# Run TensorFlow Session
# ===========================================================================

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Grab the entire validation set 
    valid_x = data_sets.validation.vcg 
    valid_y = data_sets.validation.dyssync 

    # Train the network 
    for example in range(training_iters):

        # Grab a batch and run one training step
        batch_x, batch_y = data_sets.train.next_batch(batch_size)
        _, summary = sess.run(
                        [training_step, summary_op], 
                        feed_dict={sequence: batch_x, target: batch_y})

        # Record loss for each step
        writer.add_summary(summary, step)

        # Check if we have overfitted every 50th iteration
        if step % 50 == 0:

            # Calculate the gradient norm
            norm = sess.run(gradient_norm, feed_dict={sequence: valid_x, target: valid_y})

            # Cheeck if it is above certain threshold 
            if norm < gradient_norm_threshold:
                print("Training Finished.")
                break

        # Output loss for every 10th iteration
        if step % 10 == 0:

            # Evaluate loss after one training step
            print("Loss (Iteration %d): %4f." % (step, sess.run(
                    loss, 
                    feed_dict={sequence: batch_x, target: batch_y})))

        step+=1


    # Determine accuracy
    train_x = data_sets.train.vcg 
    train_y = data_sets.train.dyssync 

    print("Training Accuracy: %2f" % sess.run(
                    accuracy, 
                    feed_dict={sequence: train_x, target: train_y}))

    print("Validation Accuracy: %2f" % sess.run(
                    accuracy, 
                    feed_dict={sequence: valid_x, target: valid_y}))

    test_x = data_sets.test.vcg 
    test_y = data_sets.test.dyssync 

    print("Testing Accuracy: %2f" % sess.run(
                    accuracy, 
                    feed_dict={sequence: test_x, target: test_y}))

