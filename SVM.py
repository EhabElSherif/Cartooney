# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:59:27 2018

@author: Dalia
"""
import numpy as np
import glob
import os
import cv2
from HOG import HOG
import pickle
from sklearn.svm import SVC

TrainingSamples = []
labels = []
model = SVC(random_state=22)

def trainSVM(Dir,Type):
    if Type !=-1:
        print("\n\nPositive Training")
    elif Type ==-1:
        print("\n\nNegative Training")

    i=0
    for filename in glob.glob(os.path.join(Dir, '*.jpg')):
        img = cv2.imread(filename, 1)
        feature = HOG(img)
        TrainingSamples.append(feature)
        labels.append(Type)
        i+=1
        if(i%1000==0):
            if Type !=-1:
                print(str(i)+" positive images are trained")
            elif Type ==-1:
                print(str(i)+" negative images are trained")    

def takeDataset():
    Dir=""
    Dir=input("Enter training data directory: ")
    while (Dir.lower()!=str("STOP").lower() ):
        Type=int(input("Enter the type of training data (1 for positive , -1 for negative): "))
        trainSVM(Dir,Type)
        print("\n\nTo break the loop and save the model ENTER STOP")
        Dir=input("Enter training data directory: ")

def fitting():
    model.fit(TrainingSamples,labels)
    print("GRAAAAAAAATZZZZ!!! Training is completed !!!!")

def saveModel():
    SVMModelName=input("Enter trained model name in format 'filename.pkl': ")
    file = open(SVMModelName, "wb")
    file.write(pickle.dumps(model))
    file.close()

takeDataset()
fitting()
saveModel()