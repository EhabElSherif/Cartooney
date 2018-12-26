# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:03:26 2018

@author: Mary
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import  threshold_adaptive,median
import sklearn.cluster as cluster
import cv2
import skimage.io as io
from skimage.color import rgb2gray


class cartoonize:
    image=[]
    cartoonizedImages=[]
    def __init__(self,img):
        self.image=img
        
    def bilateral(self):
        num_bilateral = 35
        img_color=np.copy(self.image)
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9,
                                                    sigmaColor=15,
                                                    sigmaSpace=15)
        bilateralImg=np.copy(img_color)
        self.cartoonizedImages.append(bilateralImg)
        img_gray = rgb2gray(img_color)
        img_blur = median(img_gray)
        img_edge=threshold_adaptive(img_blur,9,offset=10)
        [x,y,z]=img_color.shape

        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if(img_edge[i][j]==False):
                        img_color[i][j][k]=img_edge[i][j]
                        
        self.cartoonizedImages.append(img_color)
        
    def kmeans(self,Ks):
        n_clusters = Ks
        [x, y, z ]= self.image.shape
        img2d = self.image.reshape(x*y, z)
        Km = cluster.KMeans(n_clusters=n_clusters)
        Km.fit(img2d)
        newImgLabel = Km.fit_predict(img2d)
        segImg = np.zeros_like(img2d)
        for i in range(n_clusters):
            segImg[newImgLabel == i] = Km.cluster_centers_[i]
        segImg = segImg.reshape((x, y, z))
        self.cartoonizedImages.append(segImg)
        
    def cartooney(self):
        
        self.bilateral()
        self.kmeans(5)
        self.kmeans(7)
        self.kmeans(9)
        self.kmeans(15)
        return self.cartoonizedImages
        
                
        
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
    
#image=io.imread('camila.png')     
#c=cartoonize(image)
#c.cartooney()
#show_images(c.cartoonizedImages)
        
        
        
                        
        