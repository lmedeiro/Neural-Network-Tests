import numpy as np;

class Neuron(object):
    def __init__ (self,ID,numberOfInputs=2):
        self.ID=ID;
        self.Wn=np.random.randn(numberOfInputs);
        self.bias=np.random.randn(1);
        
    def sumInputs(self,X):
        # sum inputs;
        return 0;
    
    def sigmoidOutput(self):
        
        return 0;
    
    def sigmoidPrimeOutput(self):
        
        return 0;
    
    def setWn(self,newWn):
        
        return 0;
    
    def setBias(self,newBias):
        
        return 0;
    
    def processInfo(self,X):
        
        return 0;
    