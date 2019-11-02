import numpy as np
import cv2
from skimage.restoration import estimate_sigma

def estimate_noise(img):
    return estimate_sigma(img, multichannel=True, average_sigmas=True)


def gaussian(im, sig_x, sig_y):
    """
    This function removes noise applying a gaussian filter to
    high frequencies in the fourier space.
    
    Args:
        im: image to be denoised
        
    Return:
        den_im: denoised image
        
    """
    sx,sy = np.shape(im)[:2]
    x1 = np.linspace(-1,1,sy)
    y1 = np.linspace(-1,1,sx)
    xx,yy = np.meshgrid(x1,y1)
    gaus = np.exp(-(xx**2/(2*sig_x**2)+yy**2/(2*sig_y**2)))
    return gaus

def maxcon(im):
    """
    This function returns an image in uint8 format in the range 0,255.
    
    Args:
        im: image
        
    Return:
        maxcon(im): image re factorized
        
    """
    return np.uint8(255*((im-im.min())/(im.max()-im.min())))




def mask_square(im, x, y):
    mask = np.zeros((np.shape(im)[:2]))
    mask [x:-x, y:-y] = 1
    return mask





def remove_noise_ft(im):
    """
    This function removes noise applying a gaussian filter to
    high frequencies in the fourier space.
    
    Args:
        im: image to be denoised
        
    Return:
        den_im: denoised image
        
    """
    noise_sigma = estimate_noise(im)
    if noise_sigma>3:
        
        sx, sy, chan = np.shape(im)
        sig_x = sx/2000
        sig_y = sy/2000
        mask = gaussian(im, sig_x, sig_y)
        den_im = np.zeros(np.shape(im))
        for ch in range(0,chan):  
            im_f = np.fft.fftshift(np.fft.fft2(im[...,ch]))
            den_im[...,ch] = np.abs((np.fft.ifft2(im_f*mask)))
    else:
        den_im = im
    return maxcon(den_im)

