import numpy as np
import cv2
import os
from HOG import HOG
import pickle
from sklearn.svm import SVC
import glob
import time


face_model = pickle.load(open(input("Enter trained model file name in format 'filename.pkl': "), 'rb'))
eye_mouth_nose_model = pickle.load(open(input("Enter trained model file name in format 'filename.pkl': "), 'rb'))
test_data = []

print("\n\n-------------------------------------Testing-------------------------------------") 

i = 0
ImagesNames = [] 
StepSizeFace = 32

#minimum window size for face detection
minWinSizeFace = (32,32)

for filename in glob.glob(os.path.join('tests\\positive\\', '*.jpg')):
    
    ImagesNames.append(filename)
    img = cv2.imread(filename, 1)
    
    # FACE DETECTION
    # decide the window size for face detection depends on the smaller dimension of the pic
    if img.shape[0] < img.shape[1]:
        winSizeFace = (img.shape[0],img.shape[0])
    else:
        winSizeFace = (img.shape[1],img.shape[1])

    # loop if the current window size larger then the limit
    while winSizeFace >= minwinSizeFaceFace:
        y = 0
        while y + winSizeFace[0] <= img.shape[0]:
            x = 0
            while x + winSizeFace[1] <= img.shape[1]:
                TempImage = np.copy(img)
                cv2.rectangle(TempImage,(x,y),(x + winSizeFace[1],y + winSizeFace[0]),(0,0,255),2)
                cv2.imshow('window',TempImage)
                cv2.waitKey(1)
                time.sleep(0.0025)
                CroppedImage = img[y:y + winSizeFace[0],x:x + winSizeFace[1]]
                feature = HOG(CroppedImage)
                if svm_model.predict([feature]) == 1 :
                    cv2.rectangle(img,(x,y),(x + winSizeFace[1],y + winSizeFace[0]),(0,255,0),3)

                x+=StepSizeFace
            y+=StepSizeFace
        winSizeFace = (int(0.75 * winSizeFace[0]),int(0.75 * winSizeFace[1]))
    cv2.imshow('face',img)
    cv2.waitKey(0)
    i+=1