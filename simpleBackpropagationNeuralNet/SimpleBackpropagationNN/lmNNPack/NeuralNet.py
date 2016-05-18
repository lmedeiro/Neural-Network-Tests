'''
This is the NN project applied to MNIST. The initial goal is to practice NN concepts
with the MNIST lib of images. First test will be composed of simple backpropagation 
machines.

The algorithm for this simple, backpropagation, perceptron based NN is as follows:
     
     * Retrieve input files (MNIST in this case);
     * process the images into an input array;
     * Add a cell to the input array, totaling 1, in order to represent the bias;
     * Establish the number of initial hidden layers;
          // In this case we are starting with 1;
     * Based on the number of Hidden layers, set up the weight array;
          * Weight array should be something of the input, and the respective number of hidden layers;
               // This is for iteration through the array. 
          * Randomize all Ws. 
     // Now establishing an algo for 1 Hidden Layer specific case:
     
     * Set up 10 neurons; // each representing a number;
     * set up a W(sizeOfInputIMGs+1,1); // the +1 is for the bias;
     * Input Transposed times W + Bias;
     * Whichever Neuron has the highest signal wins;
     
     // Training the network;
     // Produce the back propagation;
     
     * Start running trainng Data samples;
     * Compare the result of the winning neuron with the actual result;
     * Calculate error;
     * Propagate the error back through the network in order to adjust; // Check respective error methods
     * Adjust Weights;
     * Run next sample;
     * Do this until overall testing error reaches a very low number (acceptable error);
     
     

'''
# Must remember to normalize the values;
# for example the input: X=X/np.amax(X,axis=0);
# sigmoid function is: y=1/(1+np.e(-z));
# sigmoid prime: np.exp(-z)/((1+np.exp(-z))**2);

import numpy as np;
import operator;
#import numba;
from Neuron import Neuron;
from threading import Thread;
#import multiprocessing;

#from scipy import optimize;

class NeuralNetwork(object):
    # This is the constructor;
    
    def __init__(self,numberOfHiddenLayers=1,numberOfNeuronsPerLayer=10):
        #print("NeuralNetwork Constructor called;");
        self.neuronN=[];
        self.eta=0.01; # learning rate eta;
        self.numberOfNeuronsPerLayer=numberOfNeuronsPerLayer;
        #self.Xn=0;
        for ID in range(self.numberOfNeuronsPerLayer):
            self.neuronN.append(Neuron(ID));
        #print(" number of neuron: %d"%len(self.neuronN));
        self.Xn=[];
    
    def feedForward(self,X):
        # Must feed information forward;
        self.Xn=[];
        for item in X:
            self.Xn.append(item);
        #print(len(self.Xn));
        
        t=[];
        r=0;
        for r in range(self.numberOfNeuronsPerLayer):
            t.append(Thread(target=self.processFeed, name=r,args=(r,)));
            #t.append(multiprocessing.Process(target=self.processFeed, name=r,args=(r,)));
            t[r].start();
        
        
        
        for item in t:
            item.join();
        
        #print("processed thread %d"%k);
        
        return self.neuronN;
    
    def processFeed(self,k):
    
        #self.neuronN[k].sumInputs(self.Xn);
        #self.neuronN[k].sigmoid();
        #self.neuronN[k].sigmoidPrime();
        self.neuronN[k].processInfo(self.Xn);
        
        return 0;
    
    def networkResponse(self):
        r=0;
        response={};
        for r in range(self.numberOfNeuronsPerLayer):
            response.update({ self.neuronN[r].ID : self.neuronN[self.neuronN[r].ID].y});
        #print(type(self.neuronN[r].y));
        maxResponse=max(response.iteritems(), key=operator.itemgetter(1));  
        #print("highest neuron/ response %f"%float(maxResponse));
        #print(maxResponse);
    
            
        
        return (maxResponse);
        
    def feedBack (self,expected,netResponse):
        # will feed the error;
        #self.netResponse=netResponse;
     
     
        # parallel update of the weights;
  
        r=0;
        
        t=[];
        for r in range(self.numberOfNeuronsPerLayer):
            t.append(Thread(target=self.calculateNewWn, name=r,args=(r,expected,netResponse,)));
            #t.append(multiprocessing.Process(target=self.processFeed, name=r,args=(r,)));
            t[r].start();
           
        for item in t:
            item.join();
        
        return self.error;
    
    
    def calculateNewWn(self,k,expected,netResponse):
        #netResponse[0] indicated the neuron fired with highest sigmoid;
        # netResponse[1] indicates what was fired;
        # expected to be calculated in separate threads;
        # zero -> 0.4; one -> 0.8;  Only one should be 0.8 or above;
        # the following case check to see if the current neuron has won
        # and it has a number above 0.8, which is our desired response;
        # if so, just return 0, and don't update the current neuron;
        # otherwise, keep going;
        if ( (k==expected) and (netResponse[0]==k) ):
            print("inside a correct response");
            if netResponse[1]>=0.8:
                error=0;
                self.error=error;
                return error;
            else: 
                error=self.calculateError(0.8,netResponse[1]);
                self.error=error;
        else:
            # otherwise, if it is not a winning neuron, 
            # then the response should be below 0.4, calculated the error
            # and update the neuron;
            error=self.calculateError(0.4,netResponse[1]);
            if ( (k!=expected) and (netResponse[0]==k) ):
                self.error=error;

        #print("error: %f"%error);
        A=error;
        #A=self.calculateSquareError();
        #A=A;
        A=np.multiply((A),self.neuronN[k].outputPrime);
        #print(A);
        B=self.neuronN[k].Xn;
        #print("from calculateNewWn: ");
        #print(B);
        
        C=np.multiply(A,B);
        C=np.multiply(C,self.eta);
        #print(C[500:550]);
        #print("neuron k: %d"%k);
        #print(self.neuronN[k].Wn);
        newBias=np.subtract(self.neuronN[k].bias,self.eta*A);
        newWn=np.subtract(self.neuronN[k].Wn,C);
        #print(newWn);
        self.neuronN[k].setWn(newWn);
        self.neuronN[k].setBias(newBias);
        
        #return self.neuronN[k].Wn;
        
    def calculateSquareError(self):
        
        errorSquared=0.5*(self.error)**2.0;
        
        return errorSquared;
        
    def calculateError(self,expected,response):
        error=expected-response;
        #error=0.5*(expected-response)**2.0;
        error=1.0*error;
        #error=self.error;
        
        return error;
    def getNeuronN(self,n):
        return self.neuronN[n];
        
        
    
   
    
    