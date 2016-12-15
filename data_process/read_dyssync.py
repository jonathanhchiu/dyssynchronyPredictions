import numpy as np 

dyssync = np.loadtxt("dyssync.txt")
np.save("dyssync.npy", dyssync)