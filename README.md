# Predicting the Dyssynchrony with Deep Learning

## Background:
### Dyssynchrony Index:
Value that is an indicator of patient response to Cardiac Resynchornization Therapy (CRT). Mathematically speaking, it is a real value between 0 and 1.

### Vectorcardiogram:
A recording of the magnitude and direction of the net electrical forces generated by the heart as expressed by a single vector with a moving head and a tail fixed at the origin. Each VCG example will be a 2D matrix with a row corresponding to a single timestep, and 3 columns corresponding to the ```x, y, z``` values of the coordinate of the head at a specified timestep. 

## Goal:
Our goal is to determine if (and how accurate if so) a VCG is able to predict the dyssynchrony index. We will be treating the VCG as a sequence of ```(x, y, z)``` coordinates through time and feeding it into recurrent neural network for classification based on the corresponding dyssynchrony.


### Dataset Dimensions:
We provide a (very) samll dataset in the interest of reducing computation time, as we are simply trying to provide proof of concept. Below are the dimensions of the dataset.
* 608 simulated VCGs
* maximum of 170 timesteps  
* 3 values per timestep ```(x, y, z)``` coordinates of the head of the vector
* 608 dyssynchrony indices.

## Files:
This repo contains two iPython notebooks: ```dataset_wrapper.ipynb``` and ```dyssync_predictions.ipynb```. The first is to document the process and reasoning behind the dataset wrapper provided (```dataset.py```). The second contains the setup and training of the neural network.

## Run:
This repo depends on TensorFlow and is written in a iPythonNotebook (Jupyter).

``` 
>>> git clone https://github.com/jonathanhchiu/dyssynchronyPredictions.git
>>> cd dyssynchronyPredictions
>>> virtualenv env
>>> source env/bin/activate
>>> pip install -r requirements.txt
>>> jupyter notebook
```


