import cv2
from skimage import feature, exposure
import numpy as np


def compute_hog(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # out, img = feature.hog(
    #     image,
    #     orientations=8,
    #     pixels_per_cell=(64, 64),
    #     cells_per_block=(5, 5),
    #     multichannel=False,
    #     visualize=True,
    # )

    out, img = feature.hog(
        image,
        orientations=8,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        multichannel=True,
        visualize=True,
    )
    img = exposure.rescale_intensity(img, in_range=(0, 10))

    return out, img

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
    # im = cv2.resize(im, (400, 400))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp_im = feature.local_binary_pattern(im, 8, 10, "nri_uniform")
    lbp_im1 = feature.local_binary_pattern(im, 8, 20, "nri_uniform")
    lbp_im2 = feature.local_binary_pattern(im, 8, 30, "nri_uniform")

    hist_im = np.concatenate(
        (
            np.histogram(lbp_im, bins, density=True)[0],
            np.histogram(lbp_im1, bins, density=True)[0],
            np.histogram(lbp_im2, bins, density=True)[0],
        )
    )

    return hist_im, lbp_im1
