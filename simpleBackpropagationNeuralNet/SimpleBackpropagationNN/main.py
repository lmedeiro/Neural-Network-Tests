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
NN=NeuralNet.NeuralNetwork();
t = time.time();
#xor=np.array([[0,0],[0,1],[1,0],[1,1]])
numberOfImgs=100;
numberOfCycles=200;
eList1=np.zeros(numberOfImgs);
#eList2=np.zeros(numberOfImgs);


totalError=[];
squaredError=[];
for _ in range(numberOfCycles):
    eList1=np.zeros(numberOfImgs);
    #eList2=np.zeros(numberOfImgs);
    k=0;
    y=0;
    for k in range(int(numberOfImgs)):
        NN.feedForward(imgs[k*784:(k+1)*784]);
        a=NN.networkResponse();
        #testA=np.sum(NN.getNeuronN(1).Wn[100:600]);
        #print("label: %d, netResponse: %d"%(labels[k],a));
        y=NN.feedBack(labels[k],a);
        eList1[k]=eList1[k]+(y==0);
        #print(eList1[k]);
        
        #testB=np.sum(NN.getNeuronN(1).Wn[100:600]);
        #test=np.subtract(testA,testB);
        #print(test)
        
    totalError.append(0.5*(1-float(np.sum(eList1))/float(len(eList1)))**2);
    #print("total error1 : %f"%totalError);
    #print ("k= %d"%k);
    '''
    k=0;
    while k <numberOfImgs:
        NN.feedForward(imgs[k*784:(k+1)*784]);
        a=NN.networkResponse();
        #testA=np.sum(NN.getNeuronN(1).Wn[100:600]);
        
        
        eList2[k]=eList2[k]+(NN.feedBack(labels[k],a)==0);
        
        #r=r+1;
        k=k+1;
        
        
    totalError=1-float(np.sum(eList2))/float(len(eList2));
    print("total error 2: %f"%totalError);
    '''
#print("Ending main Thread" );

elapsed = time.time() - t;
print ("elapsed time %f"%elapsed);
#for item in totalError:
#    print("total error1 : %f"%item);


k=0;

# Saving the variables to file;
for k in range(10):
    np.savetxt("neuronWeights_%d.txt"%k,NN.neuronN[k].Wn,fmt='%2.7f',);
    np.savetxt("neuronBias_%d.txt"%k,NN.neuronN[k].bias,fmt='%2.7f',);
np.savetxt("errorStorage.txt",totalError,fmt='%2.7f');




plt.plot(totalError);
plt.show();