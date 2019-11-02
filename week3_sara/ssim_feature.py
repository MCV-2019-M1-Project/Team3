from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


def compare_ssim(im1,im2):
    """
    This function compares the structural similarity between two images
    of different shape. They are reshaped to match shape.
    
    Args:
        im1: image dataset
        im2: image test
    Return:
        distan: the inverse of the SSIM. (The smaller the better.)
    """
    im1 = cv2.cvtColor(cv2.resize(im1, (400,400)), cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(cv2.resize(im2, (400,400)), cv2.COLOR_BGR2GRAY)
    distan = 1/ssim(im1,im2)
    return distan


def cut_image(mask, im):
    
    mask = mask * 1
    sx,sy= np.shape(mask) 
    sx_mid = np.int(sx/2)
    sy_mid = np.int(sy/2)
    horiz = mask[sx_mid,:] 
    verti = mask[:,sy_mid] 
    h = np.where(horiz == 1)
    v = np.where(verti == 1)    
    lx = np.min(h)
    rx = np.max(h)
    ty = np.min(v)
    by = np.max(v) 
    cut_im = im[ty:by, lx:rx, :]
#    print(lx,rx,ty,by)
    return cut_im
    