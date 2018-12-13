import cv2
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#HOG steps:
#1.  resize the image ##### DONE
#2.  calculate sobel gradients ##### DONE
#3.  divide the image into cells and calculate its HOG ##### DONE
#4.  divide the cells into blocks and normalize it ##### DONE
#5.  calculate the HOG feature vector ##### DONE

def HOG ( img ):
    
    img = cv2.resize(img,(64,64))
    img = np.int64(img)
    Gx = np.zeros(img.shape)        
    Gy = np.zeros(img.shape)

    #Sobel with window 3x1 (for Gx) and 1x3 (for Gy)
    #exclude border pixels
    Gx[:,1:-1] = img[:,2:] - img[:,:-2]
    Gy[1:-1,:] = img[2:,:] - img[:-2,:]
    
    Magnitude = np.zeros(img.shape)
    Magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    Orientation = np.zeros(img.shape)
    Orientation = (np.arctan2(Gy , Gx) * 180 / np.pi) % 180
    
    #Construct Cells with window 8x8
    #Every Cell has a histogram of oriented gradients
    HistogramX = np.linspace(10,170,9).astype("uint8")
    Cells = np.empty((img.shape[0] // 8,img.shape[1] // 8),dtype=list)
    
    for y in range(0,Orientation.shape[1] -7,8):
        for x in range(0,Orientation.shape[0] -7,8):
            h = np.zeros(HistogramX.shape)
            for j in range(y,y + 8):
                for i in range(x,x + 8):
                    CurrentAngle = Orientation[i,j] + 10
                    Index = int(CurrentAngle // 20)
                    h[(Index - 1) % 9]+=Magnitude[i,j] * (1 - ((CurrentAngle % 20) / 20))
                    h[(Index) % 9]+=Magnitude[i,j] * ((CurrentAngle % 20) / 20)
            Cells[x // 8][y // 8] = h

    #Construct Blocks with window 16x16
    #In progress
    Blocks = np.empty((Cells.shape[0] - 1,Cells.shape[1] - 1),dtype=list)

    HogVector = []
    for k in range(Cells.shape[0] - 1):
        for w in range(Cells.shape[1] - 1):
            #Calculate the normal of 2x2 cells 
            #normalize the 2x2 cells with this normal
            normal = np.sum(np.sum(Cells[k:k+2,w:w+2] ** 2))
            normal = np.sqrt(normal)
            h = Cells[k:k+2,w:w+2] / normal
            #append the values of the histograms vertical and horizontal in one vector "Featue vector"
            h = np.concatenate(h).ravel()
            h = np.concatenate(h).ravel()
            HogVector.extend(h)
    #Next step is to pass the feature vector to SVM to train
    return(HogVector)

#################################for testing#####################################
img = cv2.imread('tests//baby.png',cv2.IMREAD_GRAYSCALE)

if img is None:
   raise IOError('Unable to load image file')

HogVector=(HOG(img))
HogVectorLength=len(HogVector)
cv2.waitKey(0)
cv2.destroyAllWindows()
