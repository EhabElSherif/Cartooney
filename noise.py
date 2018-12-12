import utilities
import scipy.ndimage as sp
class Noise:

  def __init__(self, image):
    
    self.__img = image
    return
  
  #apply median filter if image has impulsive nois
  def medianFilter(self):
    return
  
  #apply gaussian filter when image has additive noise
  def Gaussian(self):
    return sp.gaussian_filter(self.__img,5)

img=utilities.io.imread("F:\\Google Drive\\CMP 2020\\3rd Year\\3A\\Image Processing\\Labs\\Lab 1 - STD\\pyramids.jpeg")
u=utilities.Util(img)
img=u.grey()
noise=Noise(img)
img=noise.Gaussian()
utilities.io.imshow(img)