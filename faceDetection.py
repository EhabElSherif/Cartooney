#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 07:57:07 2018

@author: ehabel-sherif
"""

import cv2
import numpy as np
import skimage.io as io
from skimage.color import rgb2hsv
from skimage.filters import gaussian

# Import Classifier for Face and Eye Detection
face_classifier = cv2.CascadeClassifier("haarcascades//haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascades//haarcascade_eye.xml")
mouth_classifier = cv2.CascadeClassifier("haarcascades//haarcascade_smile.xml")

if face_classifier.empty():
   raise IOError('Unable to load the face cascade classifier xml file')

if eye_classifier.empty():
   raise IOError('Unable to load the eye cascade classifier xml file')

if mouth_classifier.empty():
   raise IOError('Unable to load the mouth cascade classifier xml file')


def face_detector ( img ):
    # Convert Image to Grayscale
    #gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    #gray = np.array(gray, dtype='uint8')
    faces = face_classifier.detectMultiScale(img,1.1, 3)
    
    #No Faces
    if faces is ():
        return img

    # Given coordinates to detect face and eyes location from ROI
    for (x, y, w, h) in faces:

        cv2.rectangle(img,(w,y),(x+w,y+h),(0,0,255),2)
        #roi for eyes
        #detect eyes
        #roiE_gray = gray[y: y+h, x: x+w]
        #roiE_color = img[y: y+h, x: x+w]
        #eyes = eye_classifier.detectMultiScale (roiE_gray,1.3,7)
        #for (ex, ey, ew, eh) in eyes:
        #    cv2.rectangle(roiE_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            #roi_color = cv2.flip (roi_color, 1)
   #return roi_color
        
# =============================================================================
#         #roi of mouth
#         #detect mouth
#         roiM_gray = gray[y: y+h, x: x+w]
#         roiM_color = img[y: y+h, x: x+w]
#         mouth = mouth_classifier.detectMultiScale(roiM_gray,1.3,11)
#         for (mx, my, mw, mh) in mouth:
#             my=int(my-0.15*h)
#             cv2.rectangle(roiM_color,(mx,my),(mx+mw,my+mh),(0,255,255),2)
#
# =============================================================================
    
    cv2.imshow('img',img)
img = cv2.imread('tests//m(01-32)_gr.jpg')

if img is None:
   raise IOError('Unable to load image file')

face_detector(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
