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
from Neuron import Neuron
#from scipy import optimize;

class NeuralNetwork(object):
    # This is the constructor;
    
    def __init__(self,X=0):
        print("Constructor called;");
        #self.bias=1;
        if X==None or X==0:
            print("X=0 or NONE");
        else:
            print(X);
        
        #self.Wn=np.array([],np.uint8);
        #self.Xn=np.array([],np.uint8);
        #self.neuronN=Neuron();
        
        
        
    
    def feedForward(self,X,numberOfHiddenLayers=1,numberOfNeuronsPerLayer=10):
        # Must feed information forward;
        
        # Wn= weights for each input;
        # Wn will be different for each neuron that is at its end;
        # The weight themselves will vary between the different 
        # layers. Thus, we will have have varying numbers per layer;
        # To begin, only one layer will be executed. Once this base case is 
        # mastered, N layers will be adapted on the code. 
        self.Wn=np.random.randn(X.size,numberOfNeuronsPerLayer);
        self.Xn=X;
        self.neuronN=[];
        # bias will be denoted as N=number of Neurons per layer -> Rows;
        # vs M= number of hidden layers -> columns;
        self.bias=np.random.randn(numberOfNeuronsPerLayer);
        
        for k in range(numberOfNeuronsPerLayer):
            self.neuronN.append(Neuron(self.Xn,self.Wn[:,k],self.bias[k]))
            self.neuronN[k].sumInputs();
            self.neuronN[k].sigmoid();
        
            
        
        return self.neuronN;
    
    def feedBack (self,error):
        # will feed the error;
        return 0;
    
    
    
          


N=NeuralNetwork([2,3,4]);    
print ("working");    
    
    
    
    