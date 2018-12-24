import numpy as np
import cv2
from HOG import HOG
import pickle
from sklearn.svm import SVC
import time

# Check duplicated boxes inside each other
def validBox ( Boxes,p0,p1 ):
    for box in Boxes:
        if(p0[0] >= box[0] and p0[1] >= box[1] and p1[0] <= box[2] and p1[1] <= box[3]):
            return False
    return True





# Trained faces model
FaceModelFilename = 'faces.pkl'
while(FaceModelFilename == ""):
    FaceModelFilename = input("Enter face trained model file name in format 'filename.pkl': ")

# Trained nose model
NoseModelFilename = 'noses.pkl'
while(NoseModelFilename == ""):
    NoseModelFilename = input("Enter eye_mouth_nose trained model file name in format 'filename.pkl': ")

FaceModel = pickle.load(open(FaceModelFilename, 'rb'))
NoseModel = pickle.load(open(NoseModelFilename, 'rb'))

print("\n\n-------------------------------------Testing-------------------------------------") 

ImageDir = input("Enter the test image directory: ") #E:\CMP\Projects\Third Year\Cartooney\Cartooney\tests\positive\images (3).jpg
while ImageDir != "stop":
    
    # Used for NMS
    BoxesFace = []
    BoxesNose = []
    img = cv2.imread(ImageDir, 1)
    if img is None:
        print("Invalid directory !!")
        ImageDir = input("Enter the test image directory: ")
        continue
    BreakDetection = False

    
    # Resize too large images to speed up the detection process
    while (img.shape[0] > 720):
        img = cv2.resize(img,(int(np.ceil(0.8 * img.shape[1])),720))
    while (img.shape[1] > 720):
        img = cv2.resize(img,(720,int(np.ceil(0.8 * img.shape[0]))))

    # Face detection
    # decide the window size for face detection depends on the smaller
    # dimension of the pic
    # minimum window size for face detection
    if img.shape[0] < img.shape[1]:
        minWinSizeFace= max((32,32),(int(np.ceil(0.1*img.shape[0])),int(np.ceil(0.1*img.shape[0]))))
        winSizeFace = (img.shape[0],img.shape[0])
    else:
        minWinSizeFace= max((32,32),(int(np.ceil(0.1*img.shape[1])),int(np.ceil(0.1*img.shape[1]))))
        winSizeFace = (img.shape[1],img.shape[1])
        
    print("Minimum window size for face",minWinSizeFace)

    # Loop untill the current window size smaller then the limit
    while not BreakDetection and winSizeFace >= minWinSizeFace:
        print("Current window size for face",winSizeFace)

        # Calculate the step size of the sliding window depends on current
        # window size
        StepSizeFace = max(1,int(np.ceil(0.2 * min(winSizeFace[1],winSizeFace[0]))))

        yf = 0
        while not BreakDetection and yf + winSizeFace[0] <= img.shape[0]:
            xf = 0
            while not BreakDetection and xf + winSizeFace[1] <= img.shape[1]:

                # To show the sliding window
                TempImageFace = np.copy(img)
                cv2.rectangle(TempImageFace,(xf,yf),(xf + winSizeFace[1],yf + winSizeFace[0]),(0,0,255),1)
                cv2.imshow('window',TempImageFace)
                cv2.waitKey(1)
                time.sleep(0.0025)
                #if keyboard.is_pressed('space'):
                #    BreakDetection = True
                #    break

                # Get the HOG vector of current window
                CroppedImage = img[yf:yf + winSizeFace[0],xf:xf + winSizeFace[1]]
                feature = HOG(CroppedImage)
                
                # If the current window is face and it's not inside another
                # face
                if FaceModel.predict([feature]) == 1 and (len(BoxesFace) == 0 or validBox(BoxesFace,(xf,yf),(xf + winSizeFace[1],yf + winSizeFace[0]))) :

                    BoxesFace.append([xf,yf,xf + winSizeFace[1],yf + winSizeFace[0]])
                    cv2.rectangle(img,(xf,yf),(xf + winSizeFace[1],yf + winSizeFace[0]),(0,255,0),2)

                    # Nose detection
                    # Max size for window to detect nose depends on the face
                    # window size
                    maxWinSizeNose = (int(np.ceil(0.5 * winSizeFace[0])),int(np.ceil(0.3 * winSizeFace[1])))
                    winSizeNose = max((1,1),(int(np.ceil(0.3 * winSizeFace[0])),int(np.ceil(0.2 * winSizeFace[1]))))
                    
                    NoseFound = False
                    
                    # Loop untill the current window size smaller then the
                    # limit
                    while (not BreakDetection) and (not NoseFound) and winSizeNose < maxWinSizeNose:

                        # Calculate the step size of the sliding window depends
                        # on current window size
                        StepSizeNose = max(1,int(np.ceil(0.7 * min(winSizeNose[0],winSizeNose[1]))))

                        yn = yf
                        while (not BreakDetection) and (not NoseFound) and yn + winSizeNose[0] <= yf + winSizeFace[0]:
                            xn = xf
                            while (not NoseFound) and xn + winSizeNose[1] <= xf + winSizeFace[1]:
                                
                                # To show the sliding window
                                TempImageNose = np.copy(TempImageFace)
                                cv2.rectangle(TempImageNose,(xn,yn),(xn + winSizeNose[1],yn + winSizeNose[0]),(255,255,0),1)
                                cv2.imshow('window',TempImageNose)
                                cv2.waitKey(1)
                                time.sleep(0.0025)

                                #if keyboard.is_pressed('space'):
                                #    BreakDetection = True
                                #    break
                                
                                # Get HOG vector of the current window of nose
                                CroppedNose = img[yn:yn + winSizeNose[0],xn:xn + winSizeNose[1]]
                                featureNose = HOG(CroppedNose)
                                
                                # If the current window is nose
                                if(NoseModel.predict([featureNose]) == 1):
                                    BoxesNose.append([xn,yn,xn + winSizeNose[1],yn + winSizeNose[0]])
                                    cv2.rectangle(img,(xn,yn),(xn + winSizeNose[1],yn + winSizeNose[0]),(255,100,0),1)
                                    NoseFound = True

                                xn+=StepSizeNose
                            yn+=StepSizeNose
                        
                        winSizeNose = (int(np.ceil(1.1 * winSizeNose[0])),int(np.ceil(1.1 * winSizeNose[1])))
                
                xf+=StepSizeFace
            yf+=StepSizeFace
        
        winSizeFace = (int(np.ceil(0.75 * winSizeFace[0])),int(np.ceil(0.75 * winSizeFace[1])))

    cv2.imshow('face',img)
    cv2.waitKey(0)
    ImageDir = input("Enter the test image directory: ")