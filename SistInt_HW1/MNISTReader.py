'''
    This class brings the MNIST files into the system;
    
    Below is the code in Matlab: 
         fidLabel=fopen('train-labels.idx1-ubyte','r','b');
          labelHeader=fread(fidLabel,2,'uint32')
          numberOfImagesToGet=100;
          testLabels=fread(fidLabel,numberOfImagesToGet,'uint8');
          
          fclose(fidLabel);
          
          %% Starting Image processing;
          
          %clear,close all,clc;
          fid=fopen('train-images.idx3-ubyte','r','b');
          labelHeaderIMG=fread(fid,4,'uint32');
          
          imgMultiplier=28*28*numberOfImagesToGet;
          testIMGs=fread(fid,imgMultiplier,'uint8');
          % image is 28x28 pixels;
          % rows are provided;
          % images start after 4 int32 bytes;
          % offset 0016;
          % Also, since there are 28x28=768 pixels per image;
          % this is necessary for the looping;
          imgSIZE=28*28;

'''

import numpy as np;
import matplotlib.pyplot as plt;
#import threading;
from threading import Thread;
import time;

class MNISTReader(object):
    
    def __init__(self):
        #threading.Thread.__init__(self);
        self.fileIN=[];
        self.finalFile=[];
        self.labelHeader=[];
        self.imgHeader=[];
        #self.rawIMG='';
        self.imgs=np.array([],np.uint8);
        self.rawLabels='';
        self.labels=[];
        self.images=[];
        self.position=[];
        
        
    def readFile(self):
        # takes care of reading the file;
        f=open('images/train-labels.idx1-ubyte','r');
        
        self.labelHeader.append(f.read(4));
        self.labelHeader.append(f.read(4));
        
        # Setting the labelHeaders to readable characters;
        self.labelHeader[0]=int(self.labelHeader[0].encode('hex'),16);
        self.labelHeader[1]=int(self.labelHeader[1].encode('hex'),16);
        
        self.rawLabels=(f.read(-1)); #read until end of file;
        
        # set labels to integer values;
        
        k=0;
        while (k<len(self.rawLabels)):
            
            self.labels.append(int(self.rawLabels[k].encode('hex'),16));
            k=k+1;
            
        
        
        f.close();
        # open the img test file;
        imgFile=open('images/train-images.idx3-ubyte','rb');
        
        #-------------------------------------------------
        # Setting the imgHeaders to readable characters;
        #-------------------------------------------------
        k=0;
        
        
        while k<4:
            
            self.imgHeader.append(imgFile.read(4));
            #print(k);
            self.imgHeader[k]=int(self.imgHeader[k].encode('hex'),16);
            k=k+1;
        #-------------------------------------------------
        # Reading the rest of the information as img bytes;
        self.rawIMG=imgFile.read(-1);
        
        imgFile.close();
        #-------------------------------------------------
        
        #self.processImgArray();
        
        return 0;
    
    # Returns numpy array;
    # ImgArray: complete array with imgs. This may be a list;
    # numImgs: number of expected images within that array;
        #Threads will use this to separate the file into separate imgs;
    # imgSize: Size of each img;
    
    def processImgArray(self,imgArray=None,numImgs=1,imgSize=784):
        # process the file and then return the processed file
        if imgArray==None:
            imgArray=self.rawIMG;
        k=0;
      
        #t = time.time()
        # do stuff
        
        # loop initially taking in one image;
        # size= 28*28=784;
        images=[];
        #imagesNP=np.array([],np.uint8);
        while (k<numImgs*imgSize):
        
            images.append(int(imgArray[k].encode('hex'),16));
            k=k+1;
        
        self.imgs=np.append(self.imgs,images);
        
        #elapsed = time.time() - t;
        
        #print ("elapsed time %f"%elapsed);
        
        
        return self.imgs;
    
    # Returns numpy array;
    # ImgArray: complete array with imgs. This may be a list;
    # numImgs: number of expected images within that array;
        #Threads will use this to separate the file into separate imgs;
    # imgSize: Size of each img;
    # it seems that the number of threads is affecting the number of 
    # images being stored; Must check this properly;
    def encodeImgWithThreads(self,numberOfThreads=1,numImgs=1,imgArray=None,imgSize=784):
        if numImgs==1 or numberOfThreads==1:
            return self.processImgArray(imgArray, numImgs, imgSize);
        #from threading import Thread;
        threads=[];
        imgArray=self.rawIMG[0:numImgs*imgSize];
        print("length of raw img array %d"%len(imgArray));
        #dataChunk=numImgs/numberOfThreads; # setting how much data each 
        # thread is going to process;
        k=0;
        imgChunkPositions=[];
        imgChunks=[];
        while k<numberOfThreads:
            imgChunks.append(imgArray[(k*numImgs*imgSize)/numberOfThreads:((k+1)*numImgs*imgSize)/numberOfThreads]);
            imgChunkPositions.append((k*numImgs*imgSize)/numberOfThreads);
            k=k+1;
        
        #print(imgChunkPositions);
        # need to better define these parameters so to make the thread run 
        # properly;
        # each thread should take care of a portion of the img array;
        # once this portion is taken care of, we need to have a pointer to link 
        # and stitch the img properly into the numpy array;
        # This should be done with the name parameter of the thread;
        # that way we know what each thread processed (encoded);
        k=0;
        a=0;
        numberOfimages=len(imgChunks[k])/imgSize;
        while k<numberOfThreads:
            #print("number of images= %d"%(len(imgChunks[k])/imgSize));
            a=Thread(target=self.processImgArray,name=k,args=(imgChunks[k], numberOfimages, imgSize));
            a.start();
            threads.append(a);
            #threads[k].start();
            #print(threads[k].name);
            #threads[k].join();
            k=k+1;
        
        k=0;
        #threading.Lock();
        while k<numberOfThreads:
            threads[k].join();
            
            k=k+1;
       

        return self.imgs;
        # Remember that k basically associates the chunks in order;
# testing the above code: 

A=MNISTReader();
A.readFile();
#print(A.imgs.shape);

print("length of imgs: %d"%len(A.imgs));
t = time.time()
imgs=A.processImgArray(numImgs=1000);
#imgs=A.encodeImgWithThreads(numberOfThreads=2,numImgs=1000);

elapsed = time.time() - t;

print ("elapsed time %f"%elapsed);
print("length of imgs: %d"%len(A.imgs));
print("length of imgs: %d"%len(imgs));
img=imgs[998*28*28:999*28*28];
img.shape=(28,28);
plt.imshow(img,cmap='gray');
plt.show();

