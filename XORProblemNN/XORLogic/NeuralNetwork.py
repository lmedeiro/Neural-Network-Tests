import numpy as np;
from Neuron import Neuron;

class NeuralNetwork(object):
    
    def __init__(self,numberOfInputItems=2,numberOfNeuronsOutput=1,numberOfHiddenLayers=1,numberOfNeuronsPerHiddenLayer=2):
        ID=0;
        self.numberOfInputItems=numberOfInputItems;
        self.numberOfNeuronsOutput=numberOfNeuronsOutput;
        self.numberOfHiddenLayers=numberOfHiddenLayers;
        self.numberOfNeuronsPerHiddenLayer=numberOfNeuronsPerHiddenLayer;
        self.eta=0.1;
        self.neuronN=[];
        # treating input first;
        self.neuronN.append([]);
        for neurons in range (numberOfNeuronsPerHiddenLayer):
            self.neuronN[0].append(Neuron(ID,numberOfInputItems));
            ID=ID+1;
            
        
        # setting up the hidden Layers list;
        for layers in range(numberOfHiddenLayers-1):
            self.neuronN.append([]);
            #print("adding Hidden layers");
        

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
        
        #print((self.neuronN));
        #print ("ID: %d"%ID); 
        
        
        
        
    def feedForward(self,X):
        self.Xn=1.0*X;
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
        neuron=0;
        #print("length of neuron %d"%(len(self.neuronN)));
        for input in range(self.numberOfNeuronsPerHiddenLayer):
                    layerInput.append(self.neuronN[len(self.neuronN)-2][neuron].y);
                    neuron+=1;
        for neuron in range(self.numberOfNeuronsOutput):
            self.neuronN[len(self.neuronN)-1][neuron].processInfo(layerInput);
        neuron=0;
        # argument to netResponse must have only one item;
        self.netResponsef(1.0*self.neuronN[len(self.neuronN)-1][neuron].y);
        
        
        return 0;
    
    def feedback(self,expected):
        
        
        error=self.networkError(expected);
        squaredError=self.calculateSquareError();
        
        error=error*(-1.0);
        neuron=0;
        # first update last layer with network error;
        deltaWOutput=(self.eta*error*self.neuronN[len(self.neuronN)-1][neuron].outputPrime)
        deltaWOutput=np.multiply(deltaWOutput,self.neuronN[len(self.neuronN)-1][neuron].Xn)
        
        newWOutput=np.subtract(self.neuronN[len(self.neuronN)-1][neuron].Wn,deltaWOutput);
        newBiasOutput=np.subtract(self.neuronN[len(self.neuronN)-1][neuron].bias,self.eta*self.error*self.neuronN[len(self.neuronN)-1][neuron].bias);
        
        self.neuronN[len(self.neuronN)-1][neuron].setWn(newWOutput);
        self.neuronN[len(self.neuronN)-1][neuron].setBias(newBiasOutput);
        #self.neuronN[len(self.neuronN)-2][neuron]
        errorPrime=error;
        # Updates the hidden layers;
        k=0; 
        while k<self.numberOfHiddenLayers:
            
            
            for neuron in range(self.numberOfNeuronsPerHiddenLayer):
                
                errorPrime=errorPrime*self.neuronN[len(self.neuronN)-k-2][neuron].outputPrime;
                # given the current configuration, this code only works for 
                # nets ending with one neuron and same number of neurons per hidden layer;
                if (len(self.neuronN)-k-1)!=(len(self.neuronN)-1):
                    wkj=self.neuronN[len(self.neuronN)-k-1][neuron].Wn[neuron];
                else:
                    wkj=self.neuronN[len(self.neuronN)-k-1][0].Wn[neuron];
                
                deltaWkj= wkj*errorPrime;
                xn=self.neuronN[len(self.neuronN)-k-2][neuron].Xn;
                
                deltaHiddenWn=(np.multiply(xn,self.eta*self.neuronN[len(self.neuronN)-k-2][neuron].outputPrime*deltaWkj));
                deltaHiddenBias=(np.multiply(1,self.eta*self.neuronN[len(self.neuronN)-k-2][neuron].outputPrime*deltaWkj));
                
                newWn=np.subtract(self.neuronN[len(self.neuronN)-k-2][neuron].Wn,deltaHiddenWn);
                newBias=np.subtract(self.neuronN[len(self.neuronN)-k-2][neuron].bias,deltaHiddenBias);
                
                self.neuronN[len(self.neuronN)-k-2][neuron].setWn(newWn);
                self.neuronN[len(self.neuronN)-k-2][neuron].setBias(newBias);
            k+=1;
        
        
        
        return 0;
        
        
        
    
    
    def netResponsef(self,response):
        self.sigmoidOut=response;
        if response>=0.5:
            self.netResponse=1;
        else:
            self.netResponse=0;
            
    def networkError(self,expected):
        
        self.error=expected-self.netResponse;
        '''
        
        if expected==0:
            expected=0.40;
            if expected>=self.sigmoidOut:
                self.error=0;
                #print("expected= %f, error: %f"%(expected,self.error));
                return self.error;
        elif expected==1:
            expected=0.60;
            if expected<=self.sigmoidOut:
                self.error=0;
                #print("expected: %f, error: %f"%(expected,self.error));
                return self.error;
        self.error=expected-self.sigmoidOut;
        #print("general expected: %f, error: %f"%(expected,self.error));
        '''
        
        return (self.error);
       
            
    def calculateSquareError(self):
        
        self.errorSquared=0.5*(self.error)**2.0;
        
        return self.errorSquared;
    
    
    def setEta(self,eta):
            self.eta=eta;
            
    