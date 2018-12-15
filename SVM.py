# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:59:27 2018

@author: Dalia
"""
import numpy as np # linear algebra
import glob
import os
import cv2
#from matplotlib import pyplot as plt
#from skimage import color
from HOG import *
#from skimage.feature import hog
#from sklearn import svm
#from sklearn.metrics import classification_report,accuracy_score


samples = []
labels = []
def trainSVM(PositivePath,NegativePath,SVMModelName):

    positive_path = PositivePath
    negative_path = NegativePath

    # Get positive samples
    for filename in glob.glob(os.path.join(positive_path, '*.jpg')):
        img = cv2.imread(filename, 1)
        hist = HOG(img)
        samples.append(hist)
        labels.append(1)

    # Get negative samples
    for filename in glob.glob(os.path.join(negative_path, '*.jpg')):
        img = cv2.imread(filename, 1)
        hist = HOG(img)
        samples.append(hist)
        labels.append(0)

    # Convert objects to Numpy Objects
    samples = np.float32(samples)
    labels = np.array(labels)


    # Shuffle Samples
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(samples))
    samples = samples[shuffle]
    labels = labels[shuffle]    

    # Create SVM classifier
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF) # cv2.ml.SVM_LINEAR
    # svm.setDegree(0.0)
    svm.setGamma(5.383)
    # svm.setCoef0(0.0)
    svm.setC(2.67)
    # svm.setNu(0.0)
    # svm.setP(0.0)
    # svm.setClassWeights(None)

    # Train
    svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
    svm.save(SVMModelName)