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


def mask_background(image, thr=128):

    eps = [45]
    for e in eps:
        mask = np.where(
            (image < (thr + e)) & (image > (thr - e)), 0, 1).astype("uint8")

    return mask[..., None]

if __name__ == "__main__":

    image = cv2.imread("data/dataset/query2/00001.jpg")[..., ::-1]
    gray_image = transform_color(image, "Gray")
    mask = mask_background(gray_image, 156)
    cv2.imshow("2", cv2.resize(image * mask, (500, 500))[..., ::-1])
    cv2.waitKey(0)
