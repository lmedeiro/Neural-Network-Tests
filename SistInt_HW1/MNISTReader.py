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
      
        t = time.time()
        # do stuff
        
        # loop initially taking in one image;
        # size= 28*28=784;
        images=[];
        imagesNP=np.array([],np.uint8);
        while (k<numImgs*imgSize):
        
            images.append(int(imgArray[k].encode('hex'),16));
            k=k+1;
        
        imagesNP=np.append(images, self.images);
        
        elapsed = time.time() - t;
        
        print ("elapsed time %f"%elapsed);
        
        
        return imagesNP;
    
    
    
    def encodeImgWithThreads(self,numberOfThreads=1,imgArray=None,numImgs=1,imgSize=784):
        if numImgs==1 or numberOfThreads==1:
            return self.processImgArray(imgArray, numImgs, imgSize);
        from threading import Thread;
        threads=[];
        dataChunk=numImgs/numberOfThreads; # setting how much data each 
        # thread is going to process;
        k=0;
        imgChunks=[];
        imgPositions=[];
        # need to better define these parameters so to make the thread run 
        # properly;
        # each thread should take care of a portion of the img array;
        # once this portion is taken care of, we need to have a pointer to link 
        # and stitch the img properly into the numpy array;
        # This should be done with the name parameter of the thread;
        # that way we know what each thread processed (encoded);
        while k<numberOfThreads:
            threads.append(Thread(target=self.processImgArray,name=k,(imgArray, numImgs, imgSize)));
            
        
        return 0;
   
# testing the above code: 

A=MNISTReader();
A.readFile();
print(A.imgs.shape);
img=A.processImgArray();
#img=A.encodeImgWithThreads();
#img.shape=(28,28);
#plt.imshow(img,cmap='gray');
#plt.show();

