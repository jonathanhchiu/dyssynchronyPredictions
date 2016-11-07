# Wrapper classes for our VCG and dyssynchrony index data. Splits entire 
# dataset to training, validation, and testing sets. Allows for the data
# to be called in batches.
#
# This file is not meant to be executed. Its classes are imported by the
# main file, "run.py".

import numpy as np 
from math import floor 

class Dataset():
    
    def __init__(self, vcg, dyssync):
        """
        Initialize container to hold NumPy matrix.

        Parameter:
            data: NumPy matrix
        """
        self.vcg = vcg                              # NumPy matrix
        self.dyssync = dyssync                      # NumPy matrix
        self.num_examples = len(self.vcg)           # Num examples
        self.index = 0                              # Index of next available

    def next_batch(self, batch_size):
        """
        Get the next batch of examples. Begin at the "self.index"th example
        and return "batch_size" number of examples. If the remaining number 
        of examples is not enough for a full batch, take from the beginning.

        Args:
            batch_size

        Return:
            batch containing "batch_size" number of examples
        """

        # Check if the amount we want exceeds what we have
        if batch_size > self.num_examples:

            # Reset index to 0
            self.index = 0

            # Return however much we have
            return [self.vcg, self.dyssync]

        # Ending index
        last = self.index + batch_size

        # Check that the range, starting at "index" does not exceed matrix
        # dimensions
        if last <= self.num_examples:

            # Update the current index
            first = self.index
            self.index = last 

            # Return the requested range of examples 
            return [self.vcg[first:last], self.dyssync[first:last]]

        # Range starting at "index" exceeds matrix dimensions, loop back
        # to the beginning
        else:

            # First half of examples, up to end of matrix
            first_vcg = self.vcg[self.index:]
            first_dyssync = self.dyssync[self.index:]

            # Determine how many more examples we need
            self.index = last - self.num_examples

            # Grab the remaining number of matrices
            second_vcg = self.vcg[:self.index]
            second_dyssync = self.dyssync[:self.index]

            # Concatenate and return
            batch_vcg = np.concatenate((first_vcg, second_vcg), axis=0)
            batch_dyssync = np.concatenate((first_dyssync, second_dyssync), 
                                                                axis=0)

            return [batch_vcg, batch_dyssync]

class VCG():
    """
    Splits a patient's VCG simulation set into the training, validation, and 
    testing set. Each of the sets is of class "dataset" that has a method 
    to iterate through the set in batches.

    Attributes:
        training
        validation
        testing
    """
    
    def __init__(self, 
        input_file, 
        output_file,
        train_size, 
        validate_size, 
        test_size,
        one_hot=False):
        """
        Split the simulation into three sets, each of class "dataset", so that
        they can be iterable.

        Parameters:
            train_size
            validate_size
            test_size
        """

        # Read in NumPy input and output matrices
        vcg = np.load(input_file)
        dyssync = np.load(output_file).astype(int)

        # Class does not check that these two have the same length
        num_examples = len(vcg)

        # Turn dyssync into one hot vector if requested
        if one_hot:
            one_hot_dyssync = np.zeros((num_examples, 5))
            one_hot_dyssync[np.arange(num_examples), dyssync] = 1
            dyssync = one_hot_dyssync

        # Ensure that sizes equal 1
        if (train_size + validate_size + test_size > 1):

            # Set default ratio to 60/20/20
            train_size = 0.6
            validate_size = 0.2
            test_size = 0.2

        # Determine cutoff indices to split the dataset
        train_index = int(floor(num_examples * train_size))
        validate_index = int(floor(num_examples * validate_size) + train_index )
        test_size = int(floor(num_examples * test_size) + validate_index)

        # Split the vcg dataset into training, validation, and testing
        train_vcg = vcg[:train_index]
        validate_vcg = vcg[train_index + 1: validate_index]
        test_vcg = vcg[validate_index + 1: test_size]

        # Split the dyssynchrony indices similarly
        train_dyssync = dyssync[:train_index]
        validate_dyssync = dyssync[train_index + 1: validate_index]
        test_dyssync = dyssync[validate_index + 1: test_size]

        # Wrap NumPy matrices as dataset classes
        self.train = Dataset(vcg=train_vcg, dyssync=train_dyssync)
        self.validation = Dataset(vcg=validate_vcg, dyssync=validate_dyssync)
        self.test = Dataset(vcg=test_vcg, dyssync=test_dyssync)

