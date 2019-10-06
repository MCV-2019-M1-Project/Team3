import os
import cv2
import pickle
import errno
import numpy as np


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_predictions(file_name, predictions):

    with open(file_name, "wb") as f:
        pickle.dump(predictions, f)


def save_mask(file_name, mask):
    if not cv2.imwrite(file_name, mask):
        raise Exception("Can't write to disk")


def transform_color(image, color_space):
    """Transforms an image rgb image to another color space

    Args:
        image (numpy.array): input rgb image
        color_space (string): desired color_space, e.g: HSV
    """

    color_transforms = {
        "HSV": cv2.COLOR_RGB2HSV,
        "LAB": cv2.COLOR_RGB2LAB,
        "YCbCr": cv2.COLOR_RGB2YCrCb,
        "Gray": cv2.COLOR_RGB2GRAY,
        "XYZ": cv2.COLOR_RGB2XYZ,
        "HLS": cv2.COLOR_RGB2HLS,
        "Luv": cv2.COLOR_RGB2Luv,
    }

    if color_space not in color_transforms:
        raise NotImplementedError(
            "Expected colorspace to be one of {}. Instead got {}".format(
                ", ".join([*color_transforms.keys()]), color_space
            )
        )

    return cv2.cvtColor(image, color_transforms[color_space])


def mask_background(img):
    """Removes background from an image given the mean background pixel value
    Attempts to remove the background from an image by computing the distance of
    every pixel with the mean background pixel value and thresholding the
    closests pixels
    Args:
        img (numpy.array): input image
    Returns:
        The image with the background pixels in black and the mask used to
        separate foreground from background
    """

    thr1 = np.abs(img - img[10, 10]).sum(2)
    thr2 = np.abs(img - img[-10, -10]).sum(2)
    thr3 = np.abs(img - img[10, -10]).sum(2)
    thr4 = np.abs(img - img[-10, 10]).sum(2)
    thr = np.dstack((thr1, thr2, thr3, thr4)).min(2)

    mask = np.where(thr > 90, 1, 0).astype("uint8")
    mask = mask * 255

    return img, mask
