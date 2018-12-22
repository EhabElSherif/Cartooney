import skimage.io as io
import skimage.color as color
import numpy as np
  
#calculates histogram
def hist(self):
    return
  
#calculates cumulative histogram
def cumHist(self):
    return
  
#converts color image to grey
def grey(self):
    sizeX=self.__img.shape[0]
    sizeY=self.__img.shape[1]
    GrayImage=np.zeros((sizeX,sizeY))
    GrayImage =GrayImage.astype("uint8")
    GrayImage[:,:]=0.3*self.__img[:,:,0]+0.59*self.__img[:,:,1]+0.11*self.__img[:,:,2]
    return GrayImage
   
#corrects image values (hint: when values are negative, exponential, float, ..etc
def norm(self):
    return