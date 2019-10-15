from scipy import ndimage
import numpy as np
import cv2
from skimage import measure


def remove_background(images):
       """
       This function removes the background from a set of input images
       Args:
              images: set of images
       Returns:
              mask: binary image of the background mask
       """
       datatype = np.uint8

       kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(datatype)

       kernel2 = np.ones((80,80))

       count = 0
       mask = []
       for im in images:
              print(count)
              img = im
              sx,sy = np.shape(im)[0:2]
              #We are going to use the saturation channel from HSV
              img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1]

              edges = cv2.Canny(img,30,30)

              #Closing to ensure edges are continuous
              edges = cv2.dilate(edges,kernel,iterations = 1)
              edges = cv2.erode(edges, kernel, iterations=1)
     
              #Filling
              filled = (ndimage.binary_fill_holes(edges)).astype(np.float64)
    
              #Opening to remove imperfections
              filled = cv2.erode(filled, kernel2, iterations = 1)
              filled = cv2.dilate(filled, kernel2, iterations = 1)
              filled = filled.astype(np.uint8)
              
              #We remove the small elements which are not pictures
              lab = measure.label(filled)
              if np.max(lab)>1:
                     fi = np.zeros(np.shape(filled))
                     for i in range (0, np.max(lab)):
                            if np.sum((lab==(i+1))*1)>300000:
                                   a = lab==(i+1)
                                   fi = a+fi
                     filled = fi  

              mask.append(filled) 
              count = count + 1 
       return mask