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
        
        #returning the imgs;
        
        return self.processImgArray();
    
    # Returns numpy array;
    # ImgArray: complete array with imgs. This may be a list;
    # numImgs: number of expected images within that array;
        #Threads will use this to separate the file into separate imgs;
    # imgSize: Size of each img;
    
    def processImgArray(self):
        # process the file and then return the processed file
        
        
        self.imgs=np.fromstring(self.rawIMG,np.uint8);
        
        return self.imgs;

t = time.time()
A=MNISTReader();


imgs=A.readFile();


elapsed = time.time() - t;

print ("elapsed time %f"%elapsed);

print("length of imgs: %d"%len(imgs));
img=imgs[59999*28*28:60000*28*28];
img.shape=(28,28);
plt.imshow(img,cmap='gray');
plt.show();

