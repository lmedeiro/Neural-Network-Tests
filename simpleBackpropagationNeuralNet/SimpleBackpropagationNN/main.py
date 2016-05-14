import numpy as np;
from lmNNPack import *;
import matplotlib.pyplot as plt;
import time;

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
NN=NeuralNet.NeuralNetwork();
t = time.time();
a=0;
test=0;

epoch1=20000;
epoch2=2*epoch1;
eList=np.zeros(epoch2);
for k in range(epoch1):
    NN.feedForward(imgs[k*784:(k+1)*784]);
    a=NN.networkResponse();
    #testA=np.sum(NN.getNeuronN(1).Wn[100:600]);
    
    
    eList[k]=eList[k]+(NN.feedBack(labels[k],a)==0);
    
    
    #testB=np.sum(NN.getNeuronN(1).Wn[100:600]);
    #test=np.subtract(testA,testB);
    #print(test)
    
totalError=1-float(np.sum(eList))/float(len(eList));
print("total error1 : %f"%totalError);
print ("k= %d"%k);
#eList=[];
while k <epoch2:
    NN.feedForward(imgs[k*784:(k+1)*784]);
    a=NN.networkResponse();
    #testA=np.sum(NN.getNeuronN(1).Wn[100:600]);
    
    
    eList[k]=eList[k]+(NN.feedBack(labels[k],a)==0);
    
    k=k+1;
    
    
totalError=1-float(np.sum(eList[epoch2-epoch1:epoch2]))/float(len(eList));
print("total error 2: %f"%totalError);

#print("Ending main Thread" );

elapsed = time.time() - t;
print ("elapsed time %f"%elapsed);

