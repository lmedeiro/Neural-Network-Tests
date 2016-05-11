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
        print("NeuralNetwork Constructor called;");
        self.neuronN=[];
        self.numberOfNeuronsPerLayer=numberOfNeuronsPerLayer;
        
        for _ in range(self.numberOfNeuronsPerLayer):
            self.neuronN.append(Neuron());
        print(" number of neuron: %d"%len(self.neuronN));
    
    def feedForward(self,X):
        # Must feed information forward;
        self.Xn=X;
        
        
        t=[];
        
        for r in range(self.numberOfNeuronsPerLayer):
            t.append(Thread(target=self.processFeed, name=r,args=(r,)));
            #t.append(multiprocessing.Process(target=self.processFeed, name=r,args=(r,)));
            t[r].start();
        
        
        
        for item in t:
            item.join();
        
        #print("processed thread %d"%k);
        
        return self.neuronN;
    
    def processFeed(self,k):
    
        self.neuronN[k].sumInputs(self.Xn);
        self.neuronN[k].sigmoid();
        
        
        return 0;
    
    def networkResponse(self):
        response={};
        for r in range(self.numberOfNeuronsPerLayer):
            response.update({r : self.neuronN[r].y});
        maxResponse=max(response.iteritems(), key=operator.itemgetter(1))[0]    
        print("highest neuron %d"%maxResponse);
        print(response);
        #print(self.neuronN[0].y);
        return maxResponse;
        
    def feedBack (self,error):
        # will feed the error;
        return 0;
    
    
    

    
    