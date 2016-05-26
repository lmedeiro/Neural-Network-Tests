import numpy as np;

class Neuron(object):
    def __init__ (self,ID,numberOfInputs=2):
        self.ID=ID;
        self.Wn=np.random.randn(numberOfInputs)*0.1;
        self.bias=np.random.randn(1)[0]*0.1;
        self.output=0;
        
    def sumInputs(self,X):
        # sum inputs;
        self.Xn=np.array(X,dtype=float);
        if np.amax(self.Xn,axis=0):
            self.Xn=np.divide(self.Xn,float(np.amax(self.Xn,axis=0)));
        
        self.output=np.vdot(self.Wn,self.Xn);
        self.output=np.add(self.output,self.bias);
        
        
        return self.output;
    
    def sigmoidOutput(self):
        self.y=1.0/(1.0+np.exp(-self.output));
        
        return self.y;
    
    def sigmoidPrimeOutput(self):
        #self.outputPrime=np.exp(-self.output)/((1+np.exp(-self.output))**2);
        self.outputPrime=self.y*(1-self.y);
        return self.outputPrime;
    
    def setWn(self,newWn):
        #print(self.Wn);
        self.Wn=newWn;
        #print(self.Wn);
        return 0;
    
    def setBias(self,newBias):
        #print(self.bias);
        self.bias=newBias;
        #print(self.bias);
        return 0;
    
    def processInfo(self,X):
        self.sumInputs(X);
        self.sigmoidOutput();
        self.sigmoidPrimeOutput();        
        
        return self.y;
    