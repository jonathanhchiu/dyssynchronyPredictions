import numpy as np 

# Load the column vector containing the dyssynchrony indices
init_x = np.load("dyssync.npy")

# Multiply elementwise by 10, floor result, subtract 5
x_scaled = np.multiply(init_x, 10)
x_floor = np.floor(x_scaled)
x = np.subtract(x_floor, 5)

# Corner case: dyssynchrony index is between [0, 0.5)
x[x < 0] = 0

# Corner case: dyssynchrony index is 1.0
x[x > 4] = 4

# Convert each element to int 
x = x.astype(int)

# Save to file 
np.save("target.npy", x)