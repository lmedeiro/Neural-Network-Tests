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
class MNISTReader(object):
    
    def __init__(self):
        self.fileIN=[];
        self.finalFile=[];
        self.labelHeader=[];
        self.imgHeader=[];
        self.rawIMG='';
        self.imgs=np.array([]);
        self.rawLabels='';
        self.labels=[];
        self.images=[];
        
    def readFile(self):
        # takes care of reading the file;
        f=open('images/train-labels.idx1-ubyte','r');
        #print (f.tell());
        self.labelHeader.append(f.read(4));
        self.labelHeader.append(f.read(4));
        #self.labels.append(f.read(1));
        #print (f.tell());
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
        imgFile=open('images/train-images.idx3-ubyte','r');
        
        
        
        n=0;
        print(n);
        # Setting the imgHeaders to readable characters;
        k=0;
        while k<4:
            
            self.imgHeader.append(imgFile.read(4));
            #print(k);
            self.imgHeader[k]=int(self.imgHeader[k].encode('hex'),16);
            k=k+1;
            
        n=imgFile.tell();
        print(n);
        #print (f.tell());
        #self.rawIMG=f.read(-1);
        #print (f.tell());
        
        
        imgFile.close();
        return 0;
    
    def processFile(self,fileToProcess):
        # process the file and then return the processed file as
        # a single array;
        # it is then up to the Neural Net to change it accordingly;
        
        return self.finalFile;
    
    
# testing the above code: 

A=MNISTReader();
A.readFile();
h1=A.labelHeader;
#h2=int(h1.encode('hex'),16)
#h3=int(A.labels[0].encode('hex'),16);
#print (h1);

print ((A.imgHeader));