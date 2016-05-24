import numpy as np;
from Neuron import Neuron;

class NeuralNetwork(object):
    
    def __init__(self,numberOfInputItems=2,numberOfNeuronsOutput=1,numberOfHiddenLayers=1,numberOfNeuronsPerHiddenLayer=2):
        ID=0;
        self.eta=0.01;
        self.neuronN=[];
        # treating input first;
        self.neuronN.append([]);
        for neurons in range (numberOfNeuronsPerHiddenLayer):
            self.neuronN[0].append(Neuron(ID,numberOfInputItems));
            ID=ID+1;
            
        #del(neurons);
        # setting up the hidden Layers list;
        for layers in range(numberOfHiddenLayers-1):
            self.neuronN.append([]);
            print("adding Hidden layers");
        

        # adding all other neurons to the hidden neurons;
        
        if numberOfHiddenLayers>1:
            for layers in range(numberOfHiddenLayers-1):
                for neurons in range (numberOfNeuronsPerHiddenLayer):
                    self.neuronN[layers].append(Neuron(ID,numberOfNeuronsPerHiddenLayer));
                    ID=ID+1;
            
        
        
        # adding output layer: 
        
        
        self.neuronN.append([]);
        
        for neurons in range(numberOfNeuronsOutput):
            self.neuronN[len(self.neuronN)-1].append(Neuron(ID,numberOfNeuronsPerHiddenLayer));
        
        #print((self.neuronN));
        #print ("ID: %d"%ID); 
        
        
        
        
    def feedForward(self,X):
        self.Xn=X;
        
#NN=NeuralNetwork();        
        
        
        
        
        