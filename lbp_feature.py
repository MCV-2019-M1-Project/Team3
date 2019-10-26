import numpy as np
import cv2
from skimage import feature 



def crop(im,per):
    if per ==0:
        crop_im = im
    else:
        sx,sy = np.shape(im)[:2]
        sx_new = np.int(sx*per)
        sy_new = np.int(sy*per)
        crop_im = im[sx_new:-sx_new, sy_new:-sy_new, :]
    return crop_im


def loc_bin_pat(im, bins=50):
    """
    Calculates the histogram of the local binary image of an input image
    Optional parameters to modify:
        - Percentage of cropped section: 0.4
        - Number of points: 4
        - Radius: 1
        - Method: uniform

    Args:
        im: 3 color image
        bins: number of bins of the histogram (10 optimal)
        
    Rrturn:
        lbp_hist: the local binary pattern histogram of the input image
    """
#    im = crop(im, 0.4)
    im = cv2.resize(im, (400,400))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[...,0]
    lbp_im = feature.local_binary_pattern(im, 8, 2, 'nri_uniform')
    hist_im = np.histogram(lbp_im, bins, density = True)[0]
    return hist_im


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
    
