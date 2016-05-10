import numpy as np;
class Neuron(object):

    def __init__(self):
        # setup the neuron here;          
        #self.output=np.array(self.output,[]);
        print('initiated neuron');
        #self.Xn=np.array(X);
        #self.Wn=np.array(W);
        
        
    
    def sumInputs(self,X):
        # take all of the inputs and sum them together;
        # bias will be denoted as N=number of Neurons per layer -> Rows;
        # vs M= number of hidden layers -> columns;
        self.Xn=np.array(X);
        self.bias=np.random.randn(1);
        self.bias=np.absolute(self.bias)*0.1;
        # Wn= weights for each input;
        # Wn will be different for each neuron that is at its end;
        # The weight themselves will vary between the different 
        # layers. Thus, we will have have varying numbers per layer;
        # To begin, only one layer will be executed. Once this base case is 
        # mastered, N layers will be adapted on the code. 
        self.Wn=np.random.randn(X.size)*.1;
        self.Wn=np.absolute(self.Wn); # keeping the whole array positive;
        
        
        
        self.output=np.vdot(self.Xn,self.Wn);
        # need some form of normalizing this result;
        # considering we have a certain number of inputs, we will
        # use that for normalizing the vector products;
        xNormFactor=self.Xn.size;
        self.output=self.output/xNormFactor;
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
    
    def stepOutput(self,desiredInput):
        # function defining the step output;
        # if sigmoid is not used, than step may be used;
        return 0;
    
    def processInfo(self):
        # process everything here;
        Yout=self.sumInputs();
        yFinal=self.sigmoid(Yout);
        return yFinal;
    
