import utilities
class Edge:

  def __init__(self, image):
  
    self.__img = image
    return
  
  #apply canny to find edges
  def canny(self):
    #convert image to grey
    GreyImage=utilities.grey(self.__img)
    #apply gaussian filter to remove noise

    return
   
  #apply sharpening to get enhanced visual effect of edges
  def sharpening(self):
    return
