import cv2
import numpy as np


def transform_color(image, color_space):

    color_transforms = {
        "HSV": cv2.COLOR_RGB2HSV,
        "LAB": cv2.COLOR_RGB2LAB,
        "YCbCr": cv2.COLOR_RGB2YCrCb,
        "Gray": cv2.COLOR_RGB2GRAY,
    }

    if color_space not in color_transforms:
        raise NotImplementedError(
            "Expected colorspace to be one of {}. Instead got {}".format(
                ", ".join([*color_transforms.keys()]), color_space
            )
        )

    return cv2.cvtColor(image, color_transforms[color_space])


def mask_background(image, thr):

    eps = [45]
    for e in eps:
        mask = np.where(
            (image < (thr + e)) & (image > (thr - e)), 0, 1).astype("uint8")

    return mask[..., None]
