import numpy as np
import cv2
from skimage import measure
from scipy import ndimage
import pickle
import glob


def remove_background1(img):
       """
       This function removes the background from an imput image
       Args:
              img: image
       Returns:
              filled: binary image of the background mask
       """
       sx, sy = np.shape(img)[:2]
       datatype = np.uint8
       
       kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(datatype)

       kernel2 = np.ones((90,90))
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
                    if np.sum((lab==(i+1))*1)>50000:
                           a = lab==(i+1)
                           fi = a+fi
              filled = fi 
       return filled
   
def coord_box(images):
       """
       This function detects the text box of a batch of images and calcucalets 
       the mask to supress it.
       USE THIS FUNCTION IF THERE IS ONLY ONE!! PAINTING PER IMAGE
       Args:
              images: set of images
       Returns:
              coord: list of list of lists containing the coordinates
              of the text box.
              
              mask: binary image in the zone for the text box
       """
       coord = []       
       count = 0
       mask = []
       ker2 = np.ones((15,9))
       for im in images:
              sx, sy = np.shape(im)[:2]
              ker = np.ones((np.int(sx/150),np.int(sy/15)))
              #use the A nd B channels of CIE LAB
              img = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
              mask1 = img[:,:,1] == 128
              mask2 = img[:,:,2] == 128

              d = np.uint8((mask1*mask2)*255)
              

              d = cv2.erode(d, ker, iterations=1)
              d = cv2.dilate(d, ker, iterations=1)
              
              lab = measure.label(d)
              if np.max(lab)>1:
                     assert( lab.max() != 0 ) # assume at least 1 CC
                     d = ((lab == np.argmax(np.bincount(lab.flat)[1:])+1)*1).astype(np.uint8)
              else:
                     d = d
              d = cv2.dilate(d, ker2, iterations=1)       

              mask.append(d)

              x,y,w,h = cv2.boundingRect(d)
              coord.append([x,y,x+w, y+h])
              count = count+1

       return [coord], mask
   



def num_obj(img):
       """
       This function determines if there is 1 or 2 elements in one image.
       Args:
              img: image
       Returns:
              n_elements: number of elements in an image.
              arg: point at which we sould divide the image in the case there
              two elements.
       """
       sx, sy = np.shape(img)[:2]
       sx_mid = np.int(sx/2)
       sy_mid = np.int(sy/2)
       
       filled = remove_background1(img)    
       lab = measure.label(filled) 
       
       if np.max(lab)>1:
              arg = np.argmax(lab[sx_mid,:])
              if arg<sy_mid:
                     arg = np.min(np.where(lab[sx_mid,:]==1))
              else:
                     arg = arg
       else:
              arg = 0
       n_elements = np.max(lab)>1              
       return n_elements, arg

    
    
    
def coord_box1(img,add):
       """
       This function detects the text box of an image and calculates the mask 
       to supress it
       USE THIS FUNCTION IF THERE IS ONLY ONE!! PAINTING PER IMAGE
       Args:
              img: image
              add: the x coordinate at which we had to cut an image with two 
              paintings.
       Returns:
              coord: list of the coordinates of the text box.
              
              mask: binary image in the zone for the text box
       """
       sx, sy = np.shape(img)[:2]
       ker = np.ones((np.int(sx/150),np.int(sy/15)))
       ker2 = np.ones((15,9))

       img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
       mask1 = img[:,:,1] == 128
       mask2 = img[:,:,2] == 128

       d = np.uint8((mask1*mask2)*255)
       d = cv2.erode(d, ker, iterations=2)
       d = cv2.dilate(d, ker, iterations=2)
       lab = measure.label(d)
       
       if np.max(lab)>1:
              assert( lab.max() != 0 ) # assume at least 1 CC
              d = ((lab == np.argmax(np.bincount(lab.flat)[1:])+1)*1).astype(np.uint8)
       else:
              d = d
              
       d = cv2.dilate(d, ker2, iterations=1)        
       mask = d

       x,y,w,h = cv2.boundingRect(d)
       coord = [x+add,y,x+w+add, y+h]

       return coord, mask



    
def coord_box2(images):
       """
       This function detects the text box of a batch of images and calculates 
       the mask to supress it.
       The function calls other functions that already caclulate the mask and 
       coordinates for individual images.
       THIS FUNCTION CAN BE USED IF THERE IS MORE THAN ONE PAINTING PER IMAGE
       Args:
              images: set of images

       Returns:
              coord: list of list of lists of the coordinates of the text box.
              
              mask: list of binary images in the zone for the text box
       """
       coord = []
       mask = []
       for im in images:
              two, arg = num_obj(im)
              if two==True:
                     cut = np.int(arg-200)
                     im1 = im[:,:cut,:]
                     l1,m1 = coord_box1(im1,0)
                     im2 = im[:,cut:,:]
                     l2,m2 = coord_box1(im2,cut)
                     coord.append([l1,l2])
                     msk = np.concatenate((m1,m2), axis = 1)
                     mask.append(msk)
              else:
                     l,m = coord_box1(im,0)
                     coord.append([l])
                     mask.append(m)
       return coord, mask      
   
    
    
def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou   





