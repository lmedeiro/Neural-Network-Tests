#from lmNNPack.NeuralNet import NeuralNetwork;
#from lmNNPack.NeuralNet import MNISTReader;
from lmNNPack import *;

class Trainer(object):
    
    def __init__(self,paramEpochs=0,trainingSamples=0):
        self.tSamples=trainingSamples;
        self.epochs=paramEpochs;
    
    '''
     this function will take care of the training;
     feeding information backward, analyzing error,
    '''
         
    def train(self, NeuralNetwork):
        self.NN=NeuralNetwork;
        
        return 0;
    
    def setupTraining(self):
        A=MNISTReader.MNISTReader();
        imgs=A.readFile();
        labels=A.getLabels();
        
        
        return 0;
        
        