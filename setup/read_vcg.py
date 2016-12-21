import numpy as np

# Initialize python list containing the vcg lengths and input 
vcg_length = []
vcg = []

for index in range(608):

	# Create filename with zero pad
	filename = 'version{:03d}.txt'.format(index + 1)

	# Read in the text file as numpy matrix
	x = np.loadtxt(filename, delimiter="\t")
	vcg.append(x)

	# Grab and store the vcg length
	vcg_length.append(x.shape[0])

# Convert Python list to NumPy array and save
np_vcg_length = np.asarray(vcg_length)
np.save("vcg_length.npy", np_vcg_length)

np_vcg = np.asarray(vcg)
np.save("vcg.npy", np_vcg)

