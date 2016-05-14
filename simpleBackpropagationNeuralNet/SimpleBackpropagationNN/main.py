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
for k in range(20):
    NN.feedForward(imgs[k*784:(k+1)*784]);
    a=NN.networkResponse();
    NN.feedBack(labels[k],a);
    #NN.networkResponse();
#NN.feedForward(imgs);
NN.networkResponse();
#print("Ending main Thread" );

elapsed = time.time() - t;
#print ("elapsed time %f"%elapsed);

