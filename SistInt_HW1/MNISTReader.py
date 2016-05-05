'''
    This class brings the MNIST files into the system;
    
    Below is the code in Matlab: 
         fidLabel=fopen('train-labels.idx1-ubyte','r','b');
          header=fread(fidLabel,2,'uint32')
          numberOfImagesToGet=100;
          testLabels=fread(fidLabel,numberOfImagesToGet,'uint8');
          
          fclose(fidLabel);
          
          %% Starting Image processing;
          
          %clear,close all,clc;
          fid=fopen('train-images.idx3-ubyte','r','b');
          headerIMG=fread(fid,4,'uint32');
          
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
from ImageColor import str2int

class MNISTReader(object):
    
    def __init__(self):
        self.fileIN=[];
        self.finalFile=[];
        self.header=[];
        self.labels=[];
        self.images=[];
        
    def readFile(self):
        # takes care of reading the file;
        f=open('images/train-labels.idx1-ubyte','r');
        self.header.append(f.read(4));
        self.header.append(f.read(3));
        self.labels.append(f.read(1));
        print(self.header);
        
        
        f.close();
        return 0;
    
    def processFile(self,fileToProcess):
        # process the file and then return the processed file as
        # a single array;
        # it is then up to the Neural Net to change it accordingly;
        
        return self.finalFile;
    
A=MNISTReader();
A.readFile();
h1=A.header[0];
h2=int(h1.encode('hex'),16)
h3=int(A.labels[0].encode('hex'),16);
print (h2);
print (h3);
#print(type(h1[1]));