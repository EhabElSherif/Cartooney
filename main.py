# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:28:34 2018

@author: Dalia
"""

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
import cv2

# Convolution:
#from scipy.signal import convolve2d
#from scipy import fftpack
import math

from skimage.util import random_noise
#from skimage.filters import median
#from skimage.feature import canny
#from face import Face

# Edges
#from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

#detector = Face()
h= [[1,1,1],
    [1,-8,1],
    [1,1,1]]

img=io.imread('tests//feat-face-5.jpg')

gray = rgb2gray(img)
#gray = np.array(gray, dtype='uint8')

gray = cv2.Canny(gray,10,20)
cv2.imshow('img',gray)

#h = sobel(img)

#io.imshow('img',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

