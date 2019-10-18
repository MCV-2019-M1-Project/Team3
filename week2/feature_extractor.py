import numpy as np
import cv2
from week2.dataloader import Dataloader
from matplotlib import pyplot as plt


def compute_histogram_1d(img, bins=256, mask=None, sqrt=False, concat=False):
    """Computes the normalized density histogram of a given array

    Args:
        img (numpy.array): array of which to compute the histogram
        bins (int, optional): number of bins for histogram
        sqrt (bool, optional): whether to square root the computed histogram
        concat (bool, optional): whether to compute and concatenate per a
                                 channel histogram

        mask (numpy.array, optional): mask to apply to the input array

    Returns:
        The computed histogram
    """

    if mask is not None:
        mask = mask.astype("bool")

    if len(img.shape) == 3:
        if concat:
            if img.shape[2] == 3:
                hist = np.array(
                    [
                        np.histogram(
                            img[..., i][mask], bins=bins, density=True
                        )[0]
                        for i in range(3)
                    ]
                )
                hist = hist.ravel()
            else:
                raise Exception("Image should have more channels")
        else:
            hist = np.histogram(img[mask], bins=bins, density=True)[0]
    else:
        hist = np.histogram(img[mask], bins=bins, density=True)[0]

    return np.sqrt(hist) if sqrt else hist


def compute_histogram_2d(img, bins=256, mask=None, sqrt=False, concat=False):
    HIST_SIZE = [bins, bins]
    HIST_RANGE = [0, 256, 0, 256]
    CHANNELS = [0, 1]
    histogram = cv2.calcHist([img], CHANNELS, mask, HIST_SIZE, HIST_RANGE)
    cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return histogram


def compute_histogram_3d(img, bins=256, mask=None, sqrt=False, concat=False):
    HIST_SIZE = [bins, bins, bins]
    HIST_RANGE = [0, 256, 0, 256, 0, 256]
    CHANNELS = [0, 1, 2]
    histogram = cv2.calcHist([img], CHANNELS, mask, HIST_SIZE, HIST_RANGE)
    cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return histogram


def compute_mr_histogram(img, resolutions=(1, 1), bins=256, mask=None, sqrt=False):
    x_splits, y_splits = resolutions
    x_len = int(img.shape[0] / x_splits)
    y_len = int(img.shape[1] / y_splits)

    histograms = []

    for i in range(x_splits):
        for j in range(y_splits):
            small_img = img[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len]
            small_mask = mask[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len]
            histograms.append(np.histogram(small_img[small_mask], bins=bins, density=True)[0])

    histograms = [np.sqrt(hist) if sqrt else hist for hist in histograms]

    return histograms


def compute_spr_histogram(img, resolutions_list, bins=256, mask=None, sqrt=False):
    histograms = []
    for resolution in resolutions_list:
        histograms += compute_mr_histogram(img, resolution, bins, mask, sqrt)

    return np.concatenate(histograms, axis=0)


def compute_histogram(histogram_type, img, resolutions_list=[(1, 1)], bins=256, mask=None, sqrt=False,
                      concat=False):
    if histogram_type == "1D":
        return compute_histogram_1d(img, bins=bins, mask=mask, sqrt=sqrt, concat=concat)
    elif histogram_type == "2D":
        return compute_histogram_2d(img, bins=bins, mask=mask, sqrt=sqrt)
    elif histogram_type == "3D":
        return compute_histogram_3d(img, bins=bins, mask=mask, sqrt=sqrt)
    elif histogram_type == "multiresolution":
        return compute_mr_histogram(img, resolutions_list[0], bins=bins, mask=mask, sqrt=sqrt)
    elif histogram_type == "pyramid":
        compute_spr_histogram(img, resolutions_list, bins=bins, mask=mask, sqrt=sqrt)
    else:
        raise NotImplemented("you must choose from histograms types ")


if __name__ == "__main__":
    dataloader = Dataloader("data/qsd1_w2/")

    sample = dataloader[2]

    simple_hist = compute_histogram_1d(sample[1], mask=(sample[1] != 0), sqrt=True)
    plt.title("{} - simple".format(sample[0]))
    plt.plot(simple_hist)
    plt.show()

    mr_hist = compute_mr_histogram(sample[1], resolutions=(2, 2), mask=(sample[1] != 0), sqrt=True)

    i = 0
    for p in mr_hist:
        plt.title("{} - mr_{}".format(sample[0], i))
        plt.plot(p)
        plt.show()
        i += 1

    spr_hist = compute_spr_histogram(sample[1], resolutions_list=[(1, 1), (2, 2), (3, 3)],
                                     mask=(sample[1] != 0), sqrt=True)
    plt.title("{} - spr".format(sample[0]))
    plt.plot(spr_hist)
    plt.show()
