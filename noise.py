import utilities
import scipy.ndimage as sp
class Noise:

  def __init__(self, image):
    
    self.__img = image
    return
  
  #apply median filter if image has impulsive nois
  def medianFilter(self):
    return sp.median_filter(self.__img)
  
  #apply gaussian filter when image has additive noise
  def Gaussian(self):
    return sp.gaussian_filter(self.__img,5)
