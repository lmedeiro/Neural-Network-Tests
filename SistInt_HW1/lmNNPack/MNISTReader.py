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
        self.labelHeader=[];
        self.imgHeader=[];

        self.imgs=np.array([],np.uint8);
        self.rawLabels='';
        self.labels=np.array([],np.uint8);

        
        
    def readFile(self):
        # takes care of reading the file and setting everything
        # to its proper array;
        f=open('lmNNPack/images/train-labels.idx1-ubyte','r');
        
        self.labelHeader.append(f.read(4));
        self.labelHeader.append(f.read(4));
        
        # Setting the labelHeaders to readable characters;
        
        self.labelHeader[0]=int(self.labelHeader[0].encode('hex'),16);
        self.labelHeader[1]=int(self.labelHeader[1].encode('hex'),16);
        
        self.rawLabels=(f.read(-1)); #read until end of file;
        
        # set labels to integer values;
        self.labels=np.fromstring(self.rawLabels,np.uint8);
  
        f.close();
        
        # open the img test file;
        imgFile=open('lmNNPack/images/train-images.idx3-ubyte','rb');
        
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
    
    # Returns numpy array with imgs;
    def processImgArray(self):
        # process the file and then return the processed file
        
        
        self.imgs=np.fromstring(self.rawIMG,np.uint8);
        
        return self.imgs;


#-------------------------------------------------
#    Area to test the class;
#-------------------------------------------------
'''
t = time.time()
A=MNISTReader();
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