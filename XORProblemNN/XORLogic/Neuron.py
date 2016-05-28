import numpy as np;
normFactor=0.1;
class Neuron(object):
    def __init__ (self,ID,numberOfInputs=2):
        self.ID=ID;
        self.Wn=np.random.randn(numberOfInputs)*normFactor;
        #self.Wn=np.divide(self.Wn,numberOfInputs);
        self.bias=np.random.randn(1)[0]*normFactor;
        self.output=0;
        self.error=0;
        
    def sumInputs(self,X):
        # sum inputs;
        self.Xn=np.array(X,dtype=float);
        
        #self.Wn=np.divide(self.Xn,len(self.Xn));
        
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
    def setError(self,error):
        self.error=error;
        
    def processInfo(self,X):
        self.sumInputs(X);
        self.sigmoidOutput();
        self.sigmoidPrimeOutput()       
        
        return self.y;
    