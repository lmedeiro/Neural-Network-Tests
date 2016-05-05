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


class NeuralNetwork(object):
     # This is the constructor;
     
     def __init__(self,X=0):
          print("Constructor called;");
          
     def feedForward(self,X):
          # Must feed informaiton forward;
          
          
          
          
          
class Neuron (object):

    def __init__(self,inputSize,X,W,bias):
        # setup the neuron here;          
        self.output=[];
        self.input=inputSize;
        self.Xn=X;
        self.Wn=W;
        self.B=bias;
    
    def sumInputs(self):
        # take all of the inputs and sum them together;
        return self.output;
    
    def sigmoid(self):
        # defining the sigmoid function which alllows output to be 
        # to be seen as a normalized 1 or -1 output;
        # 
    
        return 0;
    
    def stepOutput(self,desiredInput):
        # function defining the step output;
        # if sigmoid is not used, than step may be used;
        return 0;
    
    
    
    
    
    
    