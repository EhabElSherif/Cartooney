import cv2
from commonfunctions import *


def ReadImg(path):
    img= cv2.imread(path)
    return img
def ShowImg(images,titels=None):
    show_images(images,titels)
    return

#Black and white image
def BA (i):
    img=np.copy(i)
    gray=rgb2gray(img)
    return gray

def Blue(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range  (img.shape[1]):
            if(img[i,j,2]<(265-50)):
                img[i,j,2]=(img[i,j,2]+50)
            elif(img[i,j,2]<(265-20)):
                img[i,j,2]=(img[i,j,2]+20)
    return img

def Red(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range  (img.shape[1]):
            if(img[i,j,0]<(265-30)):
                img[i,j,0]=(img[i,j,2]+30)

    return img

#SOBEL IS BETTER
def edgeSobel(i):
    img=np.copy(i)
    gray =rgb2gray(img)
    s=sobel(gray)
    
    return s

def edgeCanny(i):
    img=np.copy(i)
    gray =rgb2gray(img)
    c=canny(gray)
    return c

def Saturated(i):
    img=np.copy(i)
    hsv=rgb2hsv(img)
    for i in range (hsv.shape[0]):
        for j in range (hsv.shape[1]):
            if (hsv[i,j,1]<(.7)):
                hsv[i,j,1]=hsv[i,j,1] + 0.3
            elif (hsv[i,j,1]<(.9)):
                hsv[i,j,1]=hsv[i,j,1] + 0.1
            elif (hsv[i,j,1]<(.95)):
                hsv[i,j,1]=hsv[i,j,1] + 0.05
            #print(hsv[i,j,1])
    img=hsv2rgb(hsv)
            
    return img

def Hue(i):
    img=np.copy(i)
    hsv=rgb2hsv(img)
    for i in range (hsv.shape[0]):
        for j in range (hsv.shape[1]):
            if (hsv[i,j,0]<(.6)):
                hsv[i,j,0]=hsv[i,j,1] + 0.4
            elif (hsv[i,j,0]<(.9)):
                hsv[i,j,0]=hsv[i,j,1] + 0.1
            elif (hsv[i,j,0]<(.95)):
                hsv[i,j,0]=hsv[i,j,1] + 0.05
            #print(hsv[i,j,1])
    img=hsv2rgb(hsv)
            
    return img

def VALUE(i):
    img=np.copy(i)
    hsv=rgb2hsv(img)
    for i in range (hsv.shape[0]):
        for j in range (hsv.shape[1]):
            #if (hsv[i,j,2]>(.6)):
            #    hsv[i,j,2]=hsv[i,j,2]- 0.2
            if (hsv[i,j,2]>(.9)):
                hsv[i,j,2]=hsv[i,j,2] - 0.1
            elif ((hsv[i,j,2]>(.95)) or (hsv[i,j,2]>0.05)):
                hsv[i,j,2]=hsv[i,j,2] - 0.05
            #print(hsv[i,j,2])
    img=hsv2rgb(hsv)
            
    return img

def NegativeSobel(i):
    img=np.copy(i)
    gray=rgb2gray
    for i in range (0,img.shape[0]):
        for j in range (0,img.shape[1]):
           # print(img[i,j])
            if(img[i,j]>.05):
                img[i,j]=0
            else:
                img[i,j]=1
    return img      


def NegativeCanny(i):
    img=np.copy(i)
    gray=rgb2gray
    for i in range (0,img.shape[0]):
        for j in range (0,img.shape[1]):
            if (img[i,j]==True):
                img[i,j]=False
            else:
                img[i,j]=True
    return img

def addFrame(f,i):
    img=np.copy(i)
    img_resize = resize(img, (int(img.shape[0] / 1.909) ,int(img.shape[1] / 1.3469)) )
    frame=np.copy(f)
    #print(img.shape)
    #print(frame.shape)
    #print(img_resize.shape)
    frame[55:img_resize.shape[0],115:img_resize.shape[1],:]= img_resize
    img=frame
    return img
#PREPROCESSING FOR ALL 4D PHOTOS (guess_spatial_dimensions(IMG)) //RETURNS DIMENSIONS
def RGB(img):
    img=rgba2rgb(img)
    return img
def FilterMemory(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+255)/2
            img[i,j,1]=(img[i,j,1]+178)/2
            img[i,j,2]=(img[i,j,2]+102)/2
    return img

def FilterRed(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+255)/2

    return img

def FilterBlue(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,2]=(img[i,j,2]+255)/2

    return img

def FilterGreen(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,1]=(img[i,j,1]+255)/2

    return img

def FilterSun(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+253)/2
            img[i,j,1]=(img[i,j,1]+253)/2
            img[i,j,2]=(img[i,j,2]+100)/2
    return img

def FilterS(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+255)/2
            img[i,j,1]=(img[i,j,1]+215)/2
            img[i,j,2]=(img[i,j,2]+10)/2
    return img

def Filter1(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+112)/2
            img[i,j,1]=(img[i,j,1]+118)/2
            img[i,j,2]=(img[i,j,2]+144)/2
    return img

def Filter2(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+210)/2
            img[i,j,1]=(img[i,j,1]+180)/2
            img[i,j,2]=(img[i,j,2]+140)/2
    return img

def Filter3(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+245)/2
            img[i,j,1]=(img[i,j,1]+222)/2
            img[i,j,2]=(img[i,j,2]+179)/2
    return img

def Filter4(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+255)/2
            img[i,j,1]=(img[i,j,1]+228)/2
            img[i,j,2]=(img[i,j,2]+181)/2
    return img

def Filter5(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+188)/2
            img[i,j,1]=(img[i,j,1]+143)/2
            img[i,j,2]=(img[i,j,2]+143)/2
    return img

def Filter6(i):
    img=np.copy(i)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            img[i,j,0]=(img[i,j,0]+25)/2
            img[i,j,1]=(img[i,j,1]+0)/2
            img[i,j,2]=(img[i,j,2]+51)/2
    return img


def Negative(i):
    gray=rgb2gray(img)
    for i in range (gray.shape[0]):
        for j in range (gray.shape[1]):  
            if(gray[i,j]<0.5):
                gray[i,j]=1
            else:
                gray[i,j]=0
    return gray
def Check(img):
    size=guess_spatial_dimensions(IMG)
    if(size==4):
        img=RGB(img)
        print(1)
    return img

def CALL(i):
    #img=ReadImg(i)
    #img_=Check(img)
    img=i
    sat=Saturated(img)
    hue=Hue(img)
    val=VALUE(img)
    black=BA(img)
    blue= Blue(img)
    red=Red(img)
    f=FilterMemory(img)
    fr=FilterRed(img)
    fb=FilterBlue(img)
    fg=FilterGreen(img)
    fsun=FilterSun(img)
    fs=FilterS(img)
    f1=Filter1(img)
    f2=Filter2(img)
    f3=Filter3(img)
    f4=Filter4(img)
    f5=Filter5(img)
    f6=Filter6(img)
    edge_c=edgeCanny(img)
    edge_s=edgeSobel(img)
    negative_s=NegativeSobel(edge_s)
    negative_c=NegativeCanny(edge_c)
    #negative=Negative(img)
    array=[sat,hue,val,black,blue,red,f,fr,fb,fg,fsun,fs,f1,f2,f3,f4,f5,f6,edge_c,edge_s,negative_c,negative_s]
    #title=["Saturation","Hue","Value","Black & White","Blue","Red","Filter","Filter red","Filter Blue","Filter Green","Filter Sun","filter s","Filter1","Filter2","Filter3","Filter4","Filter5","filter6","Edge canny","Edge Sobbel","Negative canny","Negative Sobbel"]
    return array