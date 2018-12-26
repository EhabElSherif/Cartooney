# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:19:19 2018

@author: Mary
"""
import effects as ef

import cartoonizeClass as cartoon
import skimage.io as io
import cv2
from mask import masks,read_masks
from faceDetection import faceDetector


imgpath=input("Enter your image path\t")
image=io.imread(imgpath)
imageCV=cv2.imread(imgpath)
print("Under face detection,it may take few minutes please wait...")

FN=faceDetector(imageCV)
faces=FN[0]
noses=FN[1]
originalImage=FN[2]
while True:
    imgLoop=originalImage
    imgLoopCV=originalImage
    mode=input("Enter your desired mode \n0: for cartoonization\n1: for adding effects\n2: for adding masks\t" )
    if(mode=='0'):
        c=cartoon.cartoonize(imgLoop)
        cartoonarr=c.cartooney()
        cartoon.show_images(cartoonarr)
        cartonizemode=input("Enter your desired cartoonization mode\n enter index from 0 to 5 \t" )
        output = cartoonarr[int(cartonizemode)]
        io.imshow(cartoonarr[int(cartonizemode)])
        imgLoop=output
    elif (mode=='1'):
        arr=ef.CALL(imgLoopCV)
        cartoon.show_images(arr)
        effectmode=input("Enter your desired effect\n enter index from 0 to 21 \t" )
        output = arr[int(effectmode)]
        cartoon.show_images([arr[int(effectmode)]])
        cv2.imwrite('effect.png',arr[int(effectmode)])
        imgLoopCV=output
    elif (mode=='2'):
        arr=read_masks()
        cartoon.show_images(arr)
        maskmode=input("Enter your desired effect\n enter index from 0 to 45 \t" )
        output=masks(originalImage,int(maskmode),faces,noses)
        cv2.imwrite('masked.png',output)
        originalImage=output
#show_images(c.cartoonizedImages)

# C:\Users\Ehab Rabie\Documents\Visual Studio 2017\Projects\PythonApplication1\PythonApplication1\tests\testCases\download (55).jpg