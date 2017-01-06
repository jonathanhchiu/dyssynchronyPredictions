import numpy as np
from math import floor, ceil

class Dataset():
	def __init__(self, vcg, vcg_length, target):
		"""
		Initialize container to hold numpy matrices.

		Params:
		vcg: 3D matrix containing VCG
		vcg_length: column vector containing VCG lengths
		dyssync:
		"""

		# Get the number of examples
		self.num_examples = len(vcg)

		# Create a randomize the examples
		self.randomize = np.random.permutation(self.num_examples)

		self.vcg = vcg[self.randomize]
		self.vcg_length = vcg_length[self.randomize]
		self.target = target[self.randomize]
		self.index = 0

	def next_batch(self):
		"""
		Get the next batch of examples. We fix the batch size to be 23.

		Return:
		batch containing "batch_size" number of examples
		"""
		batch_size = 23

		# Determine bounds of batch
		begin = self.index
		end = self.index + batch_size

		# Update index: reached end of dataset (size guaranteed to be multiple
		# of 32)
		if end == self.num_examples:
			self.index = 0

		# Update index: middle of dataset
		else:
			self.index = end

		return [
			self.vcg[begin:end],
			self.vcg_length[begin:end],
			self.target[begin:end]
		]



class Patient():

	def __init__(self, vcg_file, vcg_length_file, target_file):
		"""
		Read in the VCG matrix, VCG length vector, and dyssynchrony index vector,
		and define the sizes of the training and test sets.

		1 batch = 23 examples
		1 set of 1817 training examples = 79 total batches
		Training set: 1472 examples, 64 batches (~80%)
		Testing set: 345 examples, 15 batches (~20%)

		Params:
		vcg_file:
		vcg_length_file:
		dyssync_file:
		"""

		# Read in three numpy files
		vcg = np.load(vcg_file)
		vcg_length = np.load(vcg_length_file)
		target = np.load(target_file)

		# Determine number of examples: should be 1210
		num_examples = len(vcg)
		# assert num_examples == 608, "Insufficient number of examples, need 608."
		assert num_examples == 1817, "Insufficient number of examples, need 2425."

		# Determine cutoff indices to split the dataset
		train_index = 1472
		test_index = 1817

		# Split dataset: VCG
		train_vcg = vcg[:train_index]
		test_vcg = vcg[train_index:test_index]

		# VCG sequence lengths
		train_vcg_length = vcg_length[:train_index]
		test_vcg_length = vcg_length[train_index:test_index]

		# Targets
		train_target = target[:train_index]
		test_target = target[train_index:test_index]

		# Wrap numpy matrices as dataset classes
		self.train = Dataset(
			vcg=train_vcg,
			vcg_length=train_vcg_length,
			target=train_target
		)

		self.test = Dataset(
			vcg=test_vcg,
			vcg_length=test_vcg_length,
			target=test_target
		)
