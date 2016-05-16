import numpy as np;
class Neuron(object):

    def __init__(self):
        # setup the neuron here;          
        #self.output=np.array(self.output,[]);
        #print('initiated neuron');
        #self.Xn=np.array(X);
        #self.Wn=np.array(W);
        # Wn= weights for each input;
        # Wn will be different for each neuron that is at its end;
        # The weight themselves will vary between the different 
        # layers. Thus, we will have have varying numbers per layer;
        # To begin, only one layer will be executed. Once this base case is 
        # mastered, N layers will be adapted on the code.
        imgSize=784;
        #self.Wn=np.random.randn(imgSize)/np.sqrt(imgSize);
        self.Wn=np.random.randn(imgSize)*0.1;
        # the above statement also calibrates the weights;
        #self.Wn=np.absolute(self.Wn); # keeping the whole array positive;
        self.bias=np.random.randn(1)*0.05;
        self.counter=1.0;
        #self.bias=0
        #print(self.bias);
        #self.bias=np.absolute(self.bias)*0.1;
        
    
    def sumInputs(self,X):
        # take all of the inputs and sum them together;
        # bias will be denoted as N=number of Neurons per layer -> Rows;
        # vs M= number of hidden layers -> columns;
        self.Xn=np.array(X,dtype=float);
        # normalizing the input 
        self.Xn=np.divide(self.Xn,float(255));
        #self.Xn=np.divide(self.Xn,np.std(self.Xn,axis=0));
        #print("from sumInputs:");
        #print(self.Xn);
        #self.Xn=self.Xn/np.amax(self.Xn,axis=0);
        # mean subtraction: 
        #self.Xn-=np.mean(self.Xn,axis=0);
        
        self.Wn=np.divide(self.Wn,np.std(self.Wn,axis=0));
        
        
        
        
        self.output=np.vdot(self.Xn,self.Wn);
        #bias=np.array([self.bias]);
        #print(bias);
        self.output=np.add(self.output,self.bias[0]);
        #self.output=self.output/self.counter;
        #self.counter=self.counter+1;
        #print(self.bias[0]);
        #self.output=np.add(self.output,self.bias);
        # need some form of normalizing this result;
        # considering we have a certain number of inputs, we will
        # use that for normalizing the vector products;
        #xNormFactor=self.Xn.size; # must update the normalization method;
        #self.output=self.output/xNormFactor;
        '''
        print(self.output);
        a=raw_input("run further?");
        if a=='y':
            pass
        else:
            return 0;
        self.output=self.output+self.B;
        '''
        return self.output;
    
    def sigmoid(self):
        # defining the sigmoid function which alllows output to be 
        # to be seen as a normalized 1 or -1 output;
        self.y=1/(1+np.exp(-self.output));
        #print(self.y);
        return self.y;
    
    # cannot be called before sigmoid;
    def sigmoidPrime(self):
        #Gradient of sigmoid
        self.outputPrime=np.exp(-self.output)/((1+np.exp(-self.output))**2);
        return self.outputPrime;
    
    def setWn(self,newWn):
        
        self.Wn=newWn;
        return 0;
    
    def setBias(self,newBias):
        self.bias=newBias;
        return 0;
        
    def stepOutput(self,desiredInput):
        # function defining the step output;
        # if sigmoid is not used, than step may be used;
        return 0;
    
    def processInfo(self,X):
        # process everything here;
        self.sumInputs(X);
        yFinal=self.sigmoid();
        self.sigmoidPrime();
        return yFinal;
    
