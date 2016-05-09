import numpy as np;
class Neuron(object):

    def __init__(self,X,W,bias):
        # setup the neuron here;          
        self.output=[];
        self.input=0;
        self.Xn=X;
        self.Wn=W;
        self.B=bias;
    
    def sumInputs(self):
        # take all of the inputs and sum them together;
        self.output=np.dot(self.Xn,self.Wn);
        self.output=self.output+self.B;
        return self.output;
    
    def sigmoid(self):
        # defining the sigmoid function which alllows output to be 
        # to be seen as a normalized 1 or -1 output;
        self.y=1/(1+np.exp(-self.output));
        
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
    
