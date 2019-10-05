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


if __name__ == "__main__":

    image = cv2.imread("data/dataset/query2/00001.jpg")[..., ::-1]
    gray_image = transform_color(image, "Gray")
    mask = mask_background(gray_image, 156)
    cv2.imshow("2", cv2.resize(image * mask, (500, 500))[..., ::-1])
    cv2.waitKey(0)
