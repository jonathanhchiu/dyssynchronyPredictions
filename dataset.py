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
		self.vcg = vcg
		self.vcg_length = vcg_length
		self.target = target
		self.index = 0
		self.num_examples = len(self.vcg)

	def next_batch(self):
		"""
		Get the next batch of examples. We fix the batch size to be 32.

		Return:
		batch containing "batch_size" number of examples
		"""
		batch_size = 32

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
		and define the sizes of the training, validation, and test sets.
		
		We fixed the set sizes to be as follows (1 batch = 32 examples): 
		Training set: 416 examples, 13 batches (~68%)
		Validation set: 96 examples, 2 batches (~16%)
		Testing set: 96 examples, 2 batches (~16%)

		We have 608 examples total. 
		
		Params:
		vcg_file:
		vcg_length_file:
		dyssync_file:
		"""
				
		# Read in three numpy files
		vcg = np.load(vcg_file)
		vcg_length = np.load(vcg_length_file)
		target = np.load(target_file)

		# Determine number of examples: should be 608
		num_examples = len(vcg)
		assert num_examples == 608, "Insufficient number of examples, need 608."

		# Determine cutoff indices to split the dataset
		train_index = 416
		validate_index = 512
		test_index = 608

		# Split dataset: VCG
		train_vcg = vcg[:train_index]
		validate_vcg = vcg[train_index:validate_index]
		test_vcg = vcg[validate_index:test_index]

		# VCG sequence lengths
		train_vcg_length = vcg_length[:train_index]
		validate_vcg_length = vcg_length[train_index:validate_index]
		test_vcg_length = vcg_length[validate_index:test_index]

		# Targets
		train_target = target[:train_index]
		validate_target = target[train_index:validate_index]
		test_target = target[validate_index:test_index]

		# Wrap numpy matrices as dataset classes
		self.train = Dataset(
			vcg=train_vcg, 
			vcg_length=train_vcg_length, 
			target=train_target
		)

		self.validate = Dataset(
			vcg=validate_vcg, 
			vcg_length=validate_vcg_length, 
			target=validate_target
		)

		self.test = Dataset(
			vcg=test_vcg, 
			vcg_length=test_vcg_length, 
			target=test_target
		)

