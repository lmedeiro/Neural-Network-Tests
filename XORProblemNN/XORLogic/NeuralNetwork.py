import numpy as np;
from Neuron import Neuron;

class NeuralNetwork(object):
    
    def __init__(self,numberOfInputItems=2,numberOfNeuronsOutput=1,numberOfHiddenLayers=1,numberOfNeuronsPerHiddenLayer=2):
        ID=0;
        self.numberOfInputItems=numberOfInputItems;
        self.numberOfNeuronsOutput=numberOfNeuronsOutput;
        self.numberOfHiddenLayers=numberOfHiddenLayers;
        self.numberOfNeuronsPerHiddenLayer=numberOfNeuronsPerHiddenLayer;
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
            layers=1;
            while layers<numberOfHiddenLayers:
                for neurons in range (numberOfNeuronsPerHiddenLayer):
                    self.neuronN[layers].append(Neuron(ID,numberOfNeuronsPerHiddenLayer));
                    ID=ID+1;
                layers=layers+1;
        
        
        # adding output layer: 
        
        
        self.neuronN.append([]);
        
        for neurons in range(numberOfNeuronsOutput):
            self.neuronN[len(self.neuronN)-1].append(Neuron(ID,numberOfNeuronsPerHiddenLayer));
        
        print((self.neuronN));
        print ("ID: %d"%ID); 
        
        
        
        
    def feedForward(self,X):
        self.Xn=X;
        # base case with first layer:
        for neuron in range(self.numberOfNeuronsPerHiddenLayer):
            self.neuronN[0][neuron].processInfo(self.Xn);
                
        
        
        layers=1;
        while layers<self.numberOfHiddenLayers:
            
            layerInput=[];
            for input in range(self.numberOfNeuronsPerHiddenLayer):
                    layerInput.append(self.neuronN[layers-1][neuron].y);
            for neuron in range (self.numberOfNeuronsPerHiddenLayer):
                self.neuronN[layers][neuron].processInfo(layerInput);
            layers=layers+1;
                    
        # output case, last layer: 
        layerInput=[];
        for input in range(self.numberOfNeuronsPerHiddenLayer):
                    layerInput.append(self.neuronN[len(self.neuronN)-1][neuron].y);
        for neuron in range(self.numberOfNeuronsOutput):
            self.neuronN[len(self.neuronN)][neuron].processInfo(layerInput);
        
        return 0;
        
#NN=NeuralNetwork();
        

        
        
        
        
        