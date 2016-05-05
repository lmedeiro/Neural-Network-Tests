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

class MNISTReader(object):
    
    def __init__(self):
        self.fileIN=[];
        self.finalFile=[];
        
    def readFile(self,filePathName):
        # takes care of reading the file;
        return 0;
    
    def processFile(self,fileToProcess):
        # process the file and then return the processed file as
        # a single array;
        # it is then up to the Neural Net to change it accordingly;
        
        return self.finalFile;