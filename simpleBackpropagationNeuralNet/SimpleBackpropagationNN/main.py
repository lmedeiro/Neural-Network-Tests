import numpy as np;
from lmNNPack import *;
import matplotlib.pyplot as plt;
import time;
#import pickle;

'''
t = time.time()
A=MNISTReader.MNISTReader();
imgs=A.readFile();
elapsed = time.time() - t;
print ("elapsed time %f"%elapsed);
print(A.labels.shape)
print(A.labels[0:100]);1
img=imgs[59999*28*28:60000*28*28];
img.shape=(28,28);
plt.imshow(img,cmap='gray');
plt.show();
'''
A=MNISTReader.MNISTReader();
imgs=A.readFile();
labels=A.getLabels();
#print(labels[0:200]);
#print(labels.size);

t = time.time();
#xor=np.array([[0,0],[0,1],[1,0],[1,1]])
numberOfImgs=20;
numberOfCycles=50;
eList1=np.zeros(numberOfImgs);
#eList2=np.zeros(numberOfImgs);
eta=np.linspace(0.01,0.1,10)
for i in range(len(eta)):
    NN=NeuralNet.NeuralNetwork();
    NN.setEta(eta[i]);
    totalError=[];
    squaredError=[];
    totalSquaredError=[];
    counter=0;
    
    for _ in range(numberOfCycles):
        eList1=np.zeros(numberOfImgs);
        squaredError=[];
        #eList2=np.zeros(numberOfImgs);
        k=0;
        y=0;
        for k in range(int(numberOfImgs)):
            NN.feedForward(imgs[k*784:(k+1)*784]);
            sigmoidOut=NN.networkResponse();
            #print(sigmoidOut);
            #testA=np.sum(NN.getNeuronN(1).Wn[100:600]);
            #print("label: %d, netResponse: %d"%(labels[k],sigmoidOut[0]));
            NN.feedBack(labels[k],sigmoidOut);
            y=sigmoidOut[0];
            squaredError.append(NN.calculateSquareError());
            eList1[k]=eList1[k]+(y==labels[k]);
            
            #print("squared error: %f"%squaredError[k]);
            
            #testB=np.sum(NN.getNeuronN(1).Wn[100:600]);
            #test=np.subtract(testA,testB);
            #print(test)
        totalSquaredError.append(np.mean(squaredError,axis=0));
        totalError.append((1-float(np.sum(eList1))/float(len(eList1))));
        #print("total error1 : %f"%totalError);
        #print ("k= %d"%k);
        
    
    
    k=0;
    '''
    # Saving the variables to file;
    for k in range(10):
        np.savetxt("vars/neuronWeights_%d.txt"%k,NN.neuronN[k].Wn,fmt='%2.7f',);
        np.savetxt("vars/neuronBias_%d.txt"%k,NN.neuronN[k].bias,fmt='%2.7f',);
    np.savetxt("vars/errorStorage.txt",totalError,fmt='%2.7f');
    '''
    
    
    plt.subplot(2,1,1);
    plt.plot(squaredError);
    plt.title('SquareError and totalError, eta= %f'%eta[i]);
    plt.ylabel('SquareError');
    plt.subplot(2,1,2);
    plt.plot(totalError);
    plt.ylabel('totalError');
    
    plt.show();
elapsed = time.time() - t;
print ("elapsed time %f"%elapsed);