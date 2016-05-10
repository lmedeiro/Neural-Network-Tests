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
NN=NeuralNet.NeuralNetwork();
t = time.time()
for k in range(500):
    NN.feedForward(imgs[k*784:(k+1)*784]);
#NN.feedForward(imgs);
print("Ending main Thread" );

elapsed = time.time() - t;
print ("elapsed time %f"%elapsed);
#neuronOutputs=NN.neuronN[1];
#print(neuronOutputs);
#imgs=imgs[:];
#print(imgs.size);
#b=np.ones([imgs.size,1],np.uint8)*0.5;

#c=np.dot(imgs.T,b);
#cNormalized=c/imgs.size;
#print(cNormalized);
