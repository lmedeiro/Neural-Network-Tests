from XORLogic import *;
import numpy as np;
import matplotlib.pyplot as plt;
import time;


#NN=NeuralNetwork.NeuralNetwork(numberOfInputItems=784,numberOfHiddenLayers=3,numberOfNeuronsPerHiddenLayer=20,numberOfNeuronsOutput=10);
NN=NeuralNetwork.NeuralNetwork(numberOfHiddenLayers=3,numberOfNeuronsOutput=10);
X=np.array([[0,0],[0,1],[1,0],[1,1]]);
#X=np.random.randn(784);
NN.feedForward(X[2]);
