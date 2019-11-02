import errno
import pickle
import os

import cv2
import numpy as np

from skimage.restoration import estimate_sigma
from skimage import measure
from scipy import ndimage

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_predictions(file_name, predictions):
    with open(file_name, "wb") as f:
        pickle.dump(predictions, f)


def detect_denoise(im, blur_type):

    Noise_level_before = estimate_sigma(im, average_sigmas=True, multichannel=True)

    if Noise_level_before > 3.0:
        if blur_type == "GaussianBlur":
            blur_type_last = "GaussianBlur"
            im = cv2.GaussianBlur(im, (3, 3), 0)
        elif blur_type == "medianBlur":
            blur_type_last = "medianBlur"
            im = cv2.medianBlur(im, 3)
        elif blur_type == "blur":
            blur_type_last = "blur"
            im = cv2.blur(im, (3, 3))
        elif blur_type == "bilateralFilter":
            blur_type_last = "bilateralFilter"
            im = cv2.bilateralFilter(im, 7, 50, 50)
        elif blur_type == "best":
            Noise_level_after = 1000.0
            blur_type_last = "best"
            im_ori = im.copy()
            for blur_type_try in [
                "GaussianBlur",
                "medianBlur",
                "blur",
                "bilateralFilter",
            ]:
                im_try = im_ori.copy()
                im_try, Noise_level_before_try, Noise_level_after_try, blur_type_try2 = detect_denoise(
                    im_try, blur_type_try
                )
                if Noise_level_after_try < Noise_level_after:
                    im = im_try
                    Noise_level_after = Noise_level_after_try
                    blur_type_last = blur_type_try
        else:
            raise NotImplementedError("you must choose from histograms types ")
    else:
        im = im
        blur_type_last = "none"

    Noise_level_after = estimate_sigma(im, average_sigmas=True, multichannel=True)

    return im, Noise_level_before, Noise_level_after, blur_type_last


def detect_paintings(img):
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
    sx_mid = np.int(sx / 2)
    sy_mid = np.int(sy / 2)

    image_bg = remove_bg(img)
    lab = measure.label(image_bg)

    if np.max(lab) > 1:
        split_point = np.argmax(lab[sx_mid, :])
        if split_point < sy_mid:
            split_point = np.min(np.where(lab[sx_mid, :] == 1))
        else:
            split_point = split_point
    else:
        split_point = 0
    multiple_painting = np.max(lab) > 1

    return multiple_painting, split_point, image_bg


def group_paintings(img, process_bg):

    paintings = [img]
    if process_bg:
        group, split_point, mask = detect_paintings(img)
        if group:
            add = split_point - 100
            paintings = [
                cut_image(mask[:, :add], img[:, :add, :]),
                cut_image(mask[:, add:], img[:, add:, :]),
            ]


    return paintings

def remove_bg(img):
    """
    This function removes the background from an input image
    Args:
           img: image
    Returns:
           filled: binary image of the background mask
    """
    sx, sy = np.shape(img)[:2]
    datatype = np.uint8

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(datatype)

    kernel2 = np.ones((90, 90))
    # We are going to use the saturation channel from HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]

    edges = cv2.Canny(img, 30, 30)

    # Closing to ensure edges are continuous
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Filling
    filled = (ndimage.binary_fill_holes(edges)).astype(np.float64)

    # Opening to remove imperfections
    filled = cv2.erode(filled, kernel2, iterations=1)
    filled = cv2.dilate(filled, kernel2, iterations=1)
    filled = filled.astype(np.uint8)

    # We remove the small elements which are not pictures
    lab = measure.label(filled)
    if np.max(lab) > 1:
        fi = np.zeros(np.shape(filled))
        for i in range(0, np.max(lab)):
            if np.sum((lab == (i + 1)) * 1) > 50000:
                a = lab == (i + 1)
                fi = a + fi
        filled = fi
    return filled


def cut_image(mask, im):

    mask = mask * 1
    sx, sy = np.shape(mask)
    sx_mid = np.int(sx / 2)
    sy_mid = np.int(sy / 2)
    horiz = mask[sx_mid, :]
    verti = mask[:, sy_mid]
    h = np.where(horiz == 1)
    v = np.where(verti == 1)
    lx = np.min(h)
    rx = np.max(h)
    ty = np.min(v)
    by = np.max(v)
    cut_im = im[ty:by, lx:rx, :]
    #    print(lx,rx,ty,by)
    return cut_im
