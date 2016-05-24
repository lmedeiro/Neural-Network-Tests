import numpy as np;

class Neuron(object):
    def __init__ (self,ID,numberOfInputs=2):
        self.ID=ID;
        self.Wn=np.random.randn(numberOfInputs);
        self.bias=np.random.randn(1);
        self.output=0;
        
    def sumInputs(self,X):
        # sum inputs;
        self.Xn=np.array(X,dtype=float);
        self.Xn=np.divide(self.Xn,float(np.amax(self.Xn,axis=0)));
        self.output=np.vdot(self.Wn,self.Xn);
        self.output=np.add(self.output,self.bias[0]);
        
        
        return self.output;
    
    def sigmoidOutput(self):
        self.y=1.0/(1.0+np.exp(-self.output));
        
        return self.y;
    
    def sigmoidPrimeOutput(self):
        self.outputPrime=np.exp(-self.output)/((1+np.exp(-self.output))**2);
        return self.outputPrime;
    
    def setWn(self,newWn):
        
        return 0;
    
    def setBias(self,newBias):
        
        return 0;
    
    def processInfo(self,X):
        
        return 0;
    