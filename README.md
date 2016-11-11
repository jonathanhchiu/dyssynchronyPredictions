# Predicting the Dyssynchrony with LSTMs

### Goal: 
Using VCG simulations generated from Continuity, classify the corresponding dyssynchrony index, using
a LSTM recurrent neural network implemented in TensorFlow.

### Dataset:
* 608 Simulations (More to come)
* 120 timesteps 
* 3 values per timestep (spatial coordinates)

### Run:
The program will train the neural network and output the accuracy for all three (training, validation, testing) sets at the end. The current problem is that the loss does not generally decrease; its values fluctuate cyclically. I have included TensorBoard summaries to illustrate: 

``` 
>>> git clone https://github.com/jonathanhchiu/dyssynchronyPredictions.git
>>> virtualenv env
>>> source env/bin/activate
>>> pip install -r requirements.txt
>>> python lstm.py
Loss (Iteration 10): 1.592644.
Loss (Iteration 20): 1.613625.
Loss (Iteration 30): 1.551977.
Loss (Iteration 40): 1.362071.
Loss (Iteration 50): 1.316194.
Training Accuracy: 0.414835
Validation Accuracy: 0.308333
Testing Accuracy: 0.333333
>>> tensorboard --logdir=data/
Starting TensorBoard 29 on port 6006
```
