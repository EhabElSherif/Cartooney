# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:38:09 2018

@author: Dalia
"""

import numpy as np
import cv2
import glob
import os
global masks

def read_masks():
    
    ImagesNames = [] 
    
    for filename in glob.glob(os.path.join('masks/emoj/', '*.png')):
    
        img = cv2.imread(filename, -1)
        ImagesNames.append(img)
    
    for filename in glob.glob(os.path.join('masks/mask_face/', '*.png')):
    
        img = cv2.imread(filename, -1)
        ImagesNames.append(img)
        
    for filename in glob.glob(os.path.join('masks/mask_eye/', '*.png')):
    
        img = cv2.imread(filename, -1)
        ImagesNames.append(img)
        
    for filename in glob.glob(os.path.join('masks/moustache/', '*.png')):
    
        img = cv2.imread(filename, -1)
        ImagesNames.append(img)
        
    for filename in glob.glob(os.path.join('masks/hat/', '*.png')):
    
        img = cv2.imread(filename, -1)
        ImagesNames.append(img)
        
    
    return ImagesNames




def face_detector (img):
    global masks
    
    face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    #eye_classifier = cv2.CascadeClassifier ("haarcascades/haarcascade_eye.xml")
    #mouth_classifier=cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
    nose_classifier=cv2.CascadeClassifier("haarcascades/haarcascade_nose.xml")
    
    
    
    # Convert Image to Grayscale
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')

    faces = face_classifier.detectMultiScale (gray,1.3, 3)

    if faces is ():
        return img

   
    # Given coordinates to detect face and eyes location from ROI
    for (x, y, w, h) in faces:
        
        #indecies from 0-32 mask_face 
        masked= mask(img,x,y,w,h,0,0,32)
        
        #indecies from 33-37 mask_eye
        masked= mask(img,x,y,w,h,0,0,37)
        
        #indecies from 44-47 hat
        masked= mask(img,x,y,w,h,0,0,47)
        
        roiN_gray = gray[y: y+h, x: x+w]
       
        nose = nose_classifier.detectMultiScale(roiN_gray,1.5,5)
        
        for (nx, ny, nw, nh) in nose:
            
            #indecies from 38-43 beard_moustache
            masked= mask(img,nx,ny,nw,nh,x,y,43)
        
    return masked


#adding masks to face by index  
def mask(img,nx,ny,nw,nh,x,y,index):
    
     global masks
     mask = masks[index]
     print(mask.shape)
     orig_mask = mask[:,:,3]
     orig_mask_inv = cv2.bitwise_not(orig_mask)
     mask = mask[:,:,0:3]
     origHeight, origWidth = mask.shape[:2]
    
     
     if(index < 33):
         #add mask face
        x1,y1,origW,origH = mask_face(nx,ny,nw,nh)
        
     elif(index < 38):
         #add eye mask
        x1,y1,origW,origH = mask_eye(nx,ny,nw,nh)
        
     elif(index < 44):
          #add moustache
         x1,y1,origW,origH = beard_moustache(nx,ny,nw,nh,x,y,origHeight,origWidth)
         
     else:
         #add hat on head
         x1,y1,origW,origH = hat(nx,ny,nw,nh,origHeight,origWidth)
     
     
     mask1 = cv2.resize(mask,(origW,origH),interpolation = cv2.INTER_AREA)
     mask2 = cv2.resize(orig_mask,(origW,origH),interpolation = cv2.INTER_AREA)
     mask_inv = cv2.resize(orig_mask_inv,(origW,origH),interpolation = cv2.INTER_AREA)

     roi = img[int(y1): int(y1+origH), int(x1): int(x1+origW)]

     print(mask.shape,mask1.shape,mask2.shape,roi.shape,mask_inv.shape)
     roi_bg = cv2.bitwise_and(roi,roi, mask = mask_inv)
     roi_fg = cv2.bitwise_and(mask1,mask1 ,mask = mask2)
    
     # join the roi_bg and roi_fg
     dst = cv2.add(roi_bg,roi_fg)
    
     img[int(y1): int(y1+origH), int(x1): int(x1+origW)] = dst
    
     return img


#add mask on face 
def mask_face(x,y,w,h):
    
    origW = w 
    origH = origW * h/w
    
    x1 = x #- origW/16
    x2 = x + origW #+ origW/16
    y1 = y - origH/16
    y2 = y + origH #+ origH/16
    
    # Check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
        
    origW = int(x2- x1)
    origH = int(y2 - y1)
    
    return x,y,origW,origH


#adding mask on eyes
def mask_eye(x,y,w,h):
    
    origW = w 
    origH = h
    
    return x,y,origW,origH
    

#adding moustache/beard
def beard_moustache(nx,ny,nw,nh,x,y,mh,mw):
    
    mustacheWidth = nw
    mustacheHeight = mustacheWidth * mh / mw
 
    x1 = nx - (mustacheWidth/2)
    x2 = nx + nw + (mustacheWidth/2)
    y1 = ny + nh - (mustacheHeight/5)
    y2 = ny + nh + (mustacheHeight* 3/2)
 
    # Check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
 
    # Re-calculate the width and height of the mustache image
    mustacheWidth = int(x2 - x1)
    mustacheHeight =int( y2 - y1)
 
    x1 += x 
    y1 += y
    
    return x1,y1,mustacheWidth,mustacheHeight


#adding hat on head
def hat(x,y,w,h,mh,mw):
    
    hatW = w
    hatH = hatW * mh / mw
    
    x1 = x - hatW/8
    x2 = x + w + hatW/8
    y1 = y - hatH
    y2 = y + hatH/4
    
    # Check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    
    hatW = int(x2 - x1)
    hatH = int(y2 - y1)
    
    return x1,y1,hatW,hatH


##############################################################################
    

global masks
arr = read_masks()
#print(len(arr))
masks = read_masks()

img=cv2.imread('tests/twins.jpg')

if img is None:
   raise IOError('Unable to load image file')

image = face_detector(img)

cv2.imwrite("masked.png",image)

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
        
        