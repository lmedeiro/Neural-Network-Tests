from XORLogic import *;
import numpy as np;
import matplotlib.pyplot as plt;
import time;
X=np.array([[0,0],[0,1],[1,0],[1,1]]);
D=np.array([0,1,1,0]);
epochs=500;
#eta=np.linspace(0.1,0.15,5);
eta=[0.1];
def train(item):
    NN=NeuralNetwork.NeuralNetwork(numberOfHiddenLayers=1,numberOfNeuronsPerHiddenLayer=2);
    NN.setEta(item);
    runInfo=[];
    errorToFile=[];
    totalError=[];
    totalAcertos=[];
    netSigmoidOut=[];
    for cycle in range(epochs):
        error=[];
        acertos=[];
        for turn in range(len(X)):
            index=np.random.randint(0,4);
            #index=turn;
            NN.feedForward(X[index]);
            
            NN.feedback(D[index]);
            netSigmoidOut.append(NN.sigmoidOut);
            #print("Net Response: %d"%NN.netResponse);
            #print("Net Sigmoid: %f"%NN.sigmoidOut);
            #print("X: [%d,%d], expected %d; response: %d; sigmoidOut: %f"%(X[index][0],X[index][1],D[index],NN.netResponse,NN.sigmoidOut));
            runInfo.append("X: [%d,%d], expected %d; response: %d; sigmoidOut: %f"%(X[index][0],X[index][1],D[index],NN.netResponse,NN.sigmoidOut));
            errorToFile.append(NN.error);
            error.append(NN.errorSquared);
            acertos.append(NN.netResponse==D[index]);
            
        totalAcertos.append(np.sum(acertos)/4.0);
        totalError.append(np.sum(error)/4.0);
        #print ('\n');
        #print("Net TotalErrorSquared: %f"%totalError[cycle]);
    plt.plot(totalError);
    plt.title("eta: %f"%item);
    #errorInfo=np.subtract(1,totalAcertos);
    #plt.plot(errorInfo);
    #plt.plot(totalAcertos)
    plt.show();
    np.savetxt("vars/netSigmoidOut.txt",netSigmoidOut,fmt='%2.7f',);
    np.savetxt("vars/netError.txt",errorToFile,fmt='%2.7f',);
    np.savetxt("vars/runInfo.txt",runInfo,fmt='%s',);



#NN=NeuralNetwork.NeuralNetwork(numberOfInputItems=784,numberOfHiddenLayers=3,numberOfNeuronsPerHiddenLayer=20,numberOfNeuronsOutput=10);
#NN=NeuralNetwork.NeuralNetwork(numberOfHiddenLayers=1,numberOfNeuronsPerHiddenLayer=2);

#D=np.array([1,0,0,1]);
#X=np.random.randn(784);


for item in range (len(eta)):
    train(eta[item]);

