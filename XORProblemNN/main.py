from XORLogic import *;
import numpy as np;
import matplotlib.pyplot as plt;
import time;


NN=NeuralNetwork.NeuralNetwork(numberOfHiddenLayers=3);
X=np.array([[0,0],[0,1],[1,0],[1,1]]);
NN.feedForward(X[1]);
