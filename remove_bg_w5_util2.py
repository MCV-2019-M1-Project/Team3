import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
from skimage.transform import hough_line, hough_line_peaks
import glob
import matplotlib.pylab as plt
from matplotlib import cm
import utils
import pickle
import imutils
import imageio
from sklearn import metrics
from skimage.feature import corner_peaks, corner_harris, peak_local_max, corner_fast
from skimage.filters import sobel
from PIL import Image
import PIL




def find_lines(image):
    """
    Function that computes the hough transform of an image
    Args:
        image: grayscale image
    Returns:        
        angs: array of the angles of the found lines
        distan: array of the distances of the found lines             
    """
    image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = find_contours(image)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 540)
    h, theta, d = hough_line(edges, theta=tested_angles)
    angs = np.array([])
    distan = np.array([])
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold = 0.3*np.max(h))):
        angs = np.append(angs, np.array([angle]))
        distan = np.append(distan, np.array([dist]))
        
    return angs, distan




def most_freq_angs(angs):
    """
    For a given array of detected angles it calculates the two most frequent
    It assumes that the most frequent value will be one of the ones that follow
    the sides of the painting
    Args:
        angs: Array containing all the angles found
    Returns:
        freqa: Two most frequent angles in degrees
    
    """
    hist_angs, bins = np.histogram(angs*180/np.pi, 540, [-90,90], density=True)
    #arguments of the most frequent angles
    freq_angs = np.argsort(-hist_angs)
    freq_angs_sort = np.array([freq_angs[0]]).astype(int)
    #check if the two maximums are at least 85º apart and les than 95º
    #(optimal is 90º but it's a very hard restriction)
    for i in range(1,np.shape(freq_angs)[0]-1):
        dif_angle = np.abs(bins[freq_angs[0]+1]-bins[freq_angs[i]+1])
        if dif_angle<89.5 or dif_angle>90.5:
            pass
        else:
            freq_angs_sort = np.append(freq_angs_sort, np.array([np.int(freq_angs[i]+1)]))
            break
        
    freqa = np.array([bins[freq_angs_sort[0]],bins[freq_angs_sort[1]]])
    #ored from small to big the two most frequent angles
    freqa = np.sort(freqa)
    return freqa




def find_contours(image):
    """
    Canny edge detector
    Args:
        Image: grayscale image
    Returns:
        edges: edges of the image
    """
    ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(image, ret2*0.8, ret2)
    return edges




def mask_label(image):
    """
    Computes the mask and the labeled image of an input image. It resizes the image
    a certain factor depending on the size of  the image
    Args:
        Image: grayscale image
    Return: 
        filled: binary mask of the image
        lab: labeled image of the mask. 
    """
    kernel = np.ones((5,5))
    kernel2 = np.ones((90,90))

    sx,sy = np.shape(image)[0:2]
    if sx>2000 or sy>2000:
        factor = 2
    else:
        factor = 1.5    
    
    rx = np.int(sx/factor)
    ry = np.int(sy/factor)

    image = cv2.resize(image, (ry,rx))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    canny = find_contours(image)
    canny =  cv2.dilate(canny,kernel,iterations = 1)
    canny =  cv2.erode(canny,kernel,iterations = 1)
    cnts,_ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(np.shape(canny))
    for cnt in cnts:
#        x,y,w,h = cv2.boundingRect(cnt)
#        if w>150 and h>150:
#            im = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        im = cv2.drawContours(mask,[cnt], 0, (0,255,0), 3)
#    plt.figure()
#    plt.imshow(im)
    im2 = np.ones(np.shape(canny)) - im
    im2[0:2,:] = 1
    im2[:,0:2] = 1
    im2[-3:-1 , :] = 1
    im2[:, -3:-3] = 1
    filled = (ndimage.binary_fill_holes(im2)).astype(np.float64)
    filled = cv2.erode(filled, kernel2, iterations = 1)
    filled = cv2.dilate(filled, kernel2, iterations = 1)
    filled = filled * 255
    filled = filled.astype(np.uint8)
    filled = cv2.resize(filled, (sy,sx), 0,0, interpolation = cv2.INTER_NEAREST)
    lab = np.uint8(measure.label(filled))
    lab = cv2.resize(lab, (sy,sx), 0,0, interpolation = cv2.INTER_NEAREST)
    return filled, lab




def rotation(image,label,  mask):
    """
    From a global image on the dataset it determines the rotation angle
    and rotates the image.
    Args:
        image: gray scale image
        label: labeled image of the mask
        mask: binary mask of where the paintings are
    Returns:
        rot_angle: shift of the angle. (we asume it is the same for all the paintings on the image)
        rotated_im: image rotated
        rotated_label = label rotated
        rotated_mask: mask rotated
    """
    image = np.uint8(image)
    mask = np.uint8(mask)
    
    angs, _ = find_lines(image)
    freqa = most_freq_angs(angs)

    if freqa[0]<-45:
        rot_angle = (180-freqa[1]).astype(float)
        rotated_im = imutils.rotate(image, freqa[1])
        
        label = Image.fromarray(np.uint8(label))
        rotated_label = label.rotate( -freqa[1], resample = PIL.Image.NEAREST)
        rotated_label = np.asarray(rotated_label)
        
        mask = Image.fromarray(np.uint8(mask))
        rotated_mask = mask.rotate( -freqa[1], resample = PIL.Image.NEAREST)
        rotated_mask = np.asarray(rotated_mask)
    else:
        rot_angle = np.abs(freqa[0]).astype(float)
        rotated_im = imutils.rotate(image, -rot_angle)
        
        label = Image.fromarray(np.uint8(label))
        rotated_label = label.rotate( -rot_angle, resample = PIL.Image.NEAREST)
        rotated_label = np.asarray(rotated_label)
        
        mask = Image.fromarray(np.uint8(mask))
        rotated_mask = mask.rotate( -rot_angle, resample = PIL.Image.NEAREST)
        rotated_mask = np.asarray(rotated_mask)
        
    return rot_angle, rotated_im, rotated_label, rotated_mask




def detect_paintings(label):
    """
    Given a labeled image, detects the number of paintings
    It's assumed that the image is larger on the direction where the paintings 
    are aranged
    Args:
        label: labeled image
    Returns: 
        num_paintings: number of paintings in the image
        args: Arguments of the left or bottom side of each painting
    """
    args = np.array([]).astype(int)
    sx, sy = np.shape(label)[:2]
    sxm = np.int(sx/2)
    sym = np.int(sy/2)
    num_paintings = np.max(label)
    if num_paintings==1:
        num_paintings = 1
        return num_paintings, args
    else:
       
        for i in range(1, num_paintings+1):
            if sx>sy:
                line = label[:, sym]
                arg = np.max(np.where(line==i))
                args = np.append(args, arg)
            else:
                line = label[sxm, :]
                arg = np.max(np.where(line==i))
                args = np.append(args, arg)
        args = np.sort(args)
        return num_paintings,args



def cut_painting(image, args):
    """
    This function cuts the image into the different paintings
    Args:
        image: image to cut
        args: indices where the paintings end
    Returns:
        sub_images: list of images containing a single painting
        
    """
    sx,sy = np.shape(image)[:2]
    sub_images = []
    if sx>sy:
        for i in range(0, np.shape(args)[0]):    
            if i==0:
                sub_im = image[:args[i],:]
                sub_images.append(sub_im)
            elif i == np.shape(args)[0]-1:
                sub_im = image[args[i-1]:,:]
                sub_images.append(sub_im)
            else:
                sub_im = image[args[i-1]:args[i], :]
                sub_images.append(sub_im)
    else:
        for i in range(0, np.shape(args)[0]):    
            if i==0:
                sub_im = image[:, :args[i]]
                sub_images.append(sub_im)
            elif i ==np.shape(args)[0]-1:
                sub_im = image[:, args[i-1]:]
                sub_images.append(sub_im)
                
            else:
                sub_im = image[:, args[i-1]:args[i]]
                sub_images.append(sub_im)
                
    return sub_images


def pres_rec_f1_2(GT, pred):

       recall = np.zeros(30)
       precision = np.zeros(30)
       f1 = np.zeros(30)
       count = 0
       for groundt in GT:
              sx,sy = np.shape(groundt)[:2]
              max_val = np.max(pred[count])
              if max_val == 0:
                  max_val = 1
              recall[count] = metrics.recall_score(np.reshape(groundt[:,:,0], sx*sy)/255, np.reshape(pred[count], sx*sy)/max_val)
              precision[count] = metrics.precision_score(np.reshape(groundt[:,:,0], sx*sy)/255, np.reshape(pred[count], sx*sy)/max_val)
              f1[count] = metrics.f1_score(np.reshape(groundt[:,:,0], sx*sy)/255, np.reshape(pred[count], sx*sy)/max_val)
              count = count +1
       return np.mean(precision), np.mean(recall), np.mean(f1)
   
    
images = [cv2.imread(file) for file in glob.glob("C:/Users/Sara/Datos/Master/M1/Project/week5/qsd1_w5/*.jpg")]
GTS = [cv2.imread(file) for file in glob.glob("C:/Users/Sara/Datos/Master/M1/Project/week5/qsd1_w5/*.png")]

#example on an image
im = images[5]
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
mask, label = mask_label(im)
rot_angle, rotated_im, rotated_label, rotated_mask = rotation(im, label, mask)
n_p, args = detect_paintings(rotated_label)
sub_ims = cut_painting(rotated_im, args)


c = 0
masks = []
for im in images:
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask, label = mask_label(im)
    masks.append(mask)
#    imageio.imsave('mask{}.png'.format(c), mask)
    rot_angle, rotated_im, rotated_label, rotated_mask = rotation(im, label, mask)
    imageio.imsave('rot_im{}.png'.format(c), rotated_im)
    n_p, args = detect_paintings(rotated_label)
    c = c +1

#p,r,f = pres_rec_f1_2(GTS, masks)         
#print('Precision: ', p)   
#print('Precision: ', r)
#print('Precision: ', f)
