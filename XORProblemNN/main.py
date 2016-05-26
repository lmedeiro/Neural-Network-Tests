from XORLogic import *;
import numpy as np;
import matplotlib.pyplot as plt;
import time;


#NN=NeuralNetwork.NeuralNetwork(numberOfInputItems=784,numberOfHiddenLayers=3,numberOfNeuronsPerHiddenLayer=20,numberOfNeuronsOutput=10);
NN=NeuralNetwork.NeuralNetwork(numberOfHiddenLayers=1,numberOfNeuronsPerHiddenLayer=2);
X=np.array([[0,0],[0,1],[1,0],[1,1]]);
D=np.array([0,1,1,0]);
#D=np.array([1,0,0,1]);
#X=np.random.randn(784);

epochs=1000;

totalError=[];
totalAcertos=[];
netSigmoidOut=[];
for cycle in range(epochs):
    error=[];
    acertos=[];
    for turn in range(len(X)):
        NN.feedForward(X[turn]);
        
        NN.feedback(D[turn]);
        netSigmoidOut.append(NN.sigmoidOut);
        #print("Net Response: %d"%NN.netResponse);
        #print("Net Sigmoid: %f"%NN.sigmoidOut);
        print("expected %d; response: %d; sigmoidOut: %f"%(D[turn],NN.netResponse,NN.sigmoidOut));
        
        error.append(NN.errorSquared);
        acertos.append(NN.netResponse==D[turn]);
    totalAcertos.append(np.sum(acertos)/4);
    totalError.append(np.mean(error));
    print("Net TotalErrorSquared: %f"%totalError[cycle]);
plt.plot(totalError);
#errorInfo=np.subtract(1,totalAcertos);
#plt.plot(errorInfo);
#plt.plot(totalAcertos)
plt.show();


