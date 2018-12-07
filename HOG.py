import cv2
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#HOG steps:
#1. resize the image                                        ##### DONE
#2. calculate sobel gradients                               ##### DONE
#3. divide the image into cells and calculate its HOG       ##### DONE
#4. divide the cells into blocks and normalize it                  
#5. calculate the HOG feature vector

def HOG ( img ):
    img = cv2.resize(img,(256,256))
    Gx = np.copy(img)        
    Gy = np.copy(img)

    #Sobel with window 3x1 (for Gx) and 1x3 (for Gy)
    #exclude border pixels
    Gx[:,1:-1] = img[:,2:] - img[:,:-2]
    Gy[1:-1,:] = img[2:,:] - img[:-2,:]
    
    Magnitude = np.copy(img)
    Magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    Orientation = np.copy(img)
    Orientation = np.arctan(Gy , Gx) * 180 / np.pi
    
    #Construct Cells with window 8x8
    #Every Cell has a histogram of oriented gradients
    HistogramX = np.linspace(0,160,9).astype("uint8")
    Cells = np.empty((img.shape[1] // 8,img.shape[0] // 8),dtype=list)
    
    for y in range(Orientation.shape[0] - 7):
        for x in range(Orientation.shape[1] - 7):
            h = np.zeros(HistogramX.shape)
            for j in range(y,y + 8):
                for i in range(x,x + 8):
                    index = int(Orientation[j,i] // 20)
                    #ex: Orientation = 25 with magnitude =2
                    #then 0.75 from mag[orientation] for orientation 20
                    #the rest for orientation 40

                    #if the Orientation = 20, then all mag[orientation] for orientaion 20 thats why i skip the current iteration
                    h[index]+=Magnitude[j,i] * (1 - (Orientation[j,i] - HistogramX[index]) / 20)
                    if(Orientation[j,i] - HistogramX[index]==0):
                        continue
                    h[index + 1]+=Magnitude[j,i] * (1 - (HistogramX[index + 1] - Orientation[j,i]) / 20)
            Cells[y // 8][x // 8] = h
            x+=7
        y+=7

    #Construct Blocks with window 16x16
    #In progress
    Blocks = np.empty((Cells.shape[1] // 2,Cells.shape[0] // 2),dtype=list)
    for y in range(img.shape[0] - 15):
        normal = 0
        for x in range(img.shape[1] - 15):
            for i in range(9):
                normal += Orientation[y,x][i] ** 2
            x+=15
        normal = np.sqrt(normal)
        Blocks[y][x]=normal
        y+=15