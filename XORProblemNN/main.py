from XORLogic import *;
import numpy as np;
import matplotlib.pyplot as plt;
import time;


#NN=NeuralNetwork.NeuralNetwork(numberOfInputItems=784,numberOfHiddenLayers=3,numberOfNeuronsPerHiddenLayer=20,numberOfNeuronsOutput=10);
NN=NeuralNetwork.NeuralNetwork(numberOfHiddenLayers=1);
X=np.array([[0,0],[0,1],[1,0],[1,1]]);
D=np.array([0,1,1,0]);
#X=np.random.randn(784);

epochs=200;

for cycle in range(epochs):
    for turn in range(len(X)):
        NN.feedForward(X[turn]);
        
        NN.feedback(D[turn]);
        #print("Net Response: %d"%NN.netResponse);
        #print("Net Sigmoid: %f"%NN.sigmoidOut);
        print(X[turn]);
        print(D[turn]);
    
    #print("Net errorSquared: %f"%NN.errorSquared);



