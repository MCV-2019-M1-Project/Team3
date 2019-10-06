import os
import cv2
import pickle
import errno
import numpy as np
from matplotlib import pyplot as plt

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


def estimate_background(img, ratios=[0.1, 0.15, 0.20]):
    """Estimates the mean pixel value of an image background

    Attempts to compute the mean background pixel value by taking different
    ratios of the edges of the image and averaging them together

    Args:
        img (numpy.array): input image
        ratios (list, optional): The percentage of the image you want to
        crop from the edges

    Returns:
        Array containing the mean background pixel value
    """

    mean_bgn = np.array([])
    for ratio in ratios:
        border = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
        mask = np.ones_like(img).astype("bool")
        mask[border[0] : -border[0], border[1] : -border[1]] = 0
        img = np.where(mask == 0, 0, img).reshape(-1, img.shape[-1])
        mean_bgn = np.append(mean_bgn, img.mean(0))

    mean_bgn = mean_bgn.reshape(-1, img.shape[-1]).mean(0)

    return mean_bgn


def mask_background(img, mean_bgn):
    """Removes background from an image given the mean background pixel value

    Attempts to remove the background from an image by computing the distance of
    every pixel with the mean background pixel value and thresholding the
    closests pixels

    Args:
        img (numpy.array): input image
        mean_bgn (numpy.array): the mean pixel value of the background

    Returns:
        The image with the background pixels in black and the mask used to
        separate foreground from background
    """

    pixel_norm = np.linalg.norm(img, axis=2)

    bgn_norm = np.linalg.norm(mean_bgn)
    low_thr = bgn_norm - 105
    high_thr = bgn_norm + 105
    mask = np.where(
        (pixel_norm > (high_thr)) | (pixel_norm < (low_thr)), 0, 1
    ).astype("uint8")
    mask = mask * 255

    return img, mask


def plot_histogram(histogram):

    plt.plot(histogram)
    plt.show()
