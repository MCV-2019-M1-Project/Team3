import numpy as np
import cv2
from week3.dataloader import Dataloader
from docutils.nodes import block_quote
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from week3.utils import cut_image


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


def compute_mr_histogram(img, splits=(1, 1), bins=256, mask=None, sqrt=False, concat=False):
    x_splits, y_splits = splits
    x_len = int(img.shape[0] / x_splits)
    y_len = int(img.shape[1] / y_splits)

    histograms = []

    for i in range(x_splits):
        for j in range(y_splits):
            small_img = img[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len]
            small_mask = None
            if mask is not None:
                small_mask = mask[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len].astype("bool")
            if concat:
                if len(small_img.shape) == 3:
                    small_hist = np.array([
                        np.histogram(small_img[..., channel][small_mask], bins=bins, density=True)[0]
                        for channel in range(small_img.shape[2])
                    ])
                    histograms.append(small_hist.ravel())
                else:
                    raise Exception("Image should have more channels")
            else:
                histograms.append((np.histogram(small_img[small_mask], bins=bins, density=True)[0]).ravel())

    histograms = [np.sqrt(hist) if sqrt else hist for hist in histograms]
    return np.concatenate(histograms, axis=0)


def compute_spr_histogram(img, rec_level, bins=256, mask=None, sqrt=False, concat=False):
    histograms = np.array([], )
    for resolution in range(rec_level, 0, -1):
        histograms = np.concatenate(
            (histograms, compute_mr_histogram(img, (resolution, resolution), bins, mask, sqrt, concat).ravel()))

    return histograms


def compute_feature_vector(feature_type, img, splits=(1, 1), rec_level=1, bins=256, mask=None, sqrt=False,
                           concat=False, hog_params=None, dct_block_size=32, dct_coeffs=10):
    if feature_type == "1D":
        return compute_histogram_1d(img, bins=bins, mask=mask, sqrt=sqrt, concat=concat)
    elif feature_type == "2D":
        return compute_histogram_2d(img, bins=bins, mask=mask, sqrt=sqrt)
    elif feature_type == "3D":
        return compute_histogram_3d(img, bins=bins, mask=mask, sqrt=sqrt)
    elif feature_type == "multiresolution":
        return compute_mr_histogram(img, splits, bins=bins, mask=mask, sqrt=sqrt, concat=concat)
    elif feature_type == "pyramid":
        return compute_spr_histogram(img, rec_level, bins=bins, mask=mask, sqrt=sqrt, concat=concat)
    elif feature_type == "hog":
        return compute_image_hog(img, hog_params, mask=mask)
    elif feature_type == "dct":
        return compute_image_dct(img, mask=mask, block_size=dct_block_size,num_coefs=dct_coeffs)

    else:
        raise NotImplemented("you must choose from histograms types ")


def compute_image_dct(image, block_size=64, num_coefs=10, mask=None):
    if mask is not None:
        image = cut_image(mask.astype("bool"), image)

    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dct_out = []
    for x_block in np.r_[:image.shape[0]:block_size]:
        for y_block in np.r_[:image.shape[1]:block_size]:
            dct_im = cv2.dct(image[x_block:x_block + block_size, y_block:y_block + block_size] / 255)
            dct_coffs = np.concatenate([np.diagonal(dct_im[::-1, :], i)[::(2 * (i % 2) - 1)]
                                        for i in range(1 - dct_im.shape[0], dct_im.shape[0])])[:num_coefs]

            dct_out.append(dct_coffs)

    return np.array(dct_out).ravel()


def compute_image_hog(image, hog_params, mask=None):
    if mask is not None:
        image = image[mask.astype("bool"), :].astype("float")

    image = cv2.resize(image, (512, 512))

    hog = cv2.HOGDescriptor(hog_params['window'], hog_params['block_size'], hog_params['block_stride'],
                            hog_params['cell_size'], hog_params['bins'])

    return hog.compute(image)


if __name__ == "__main__":
    dataloader = Dataloader("../data/qsd2_w3/", evaluate=True)

    hog_params = {
        "window": (128, 128),
        "block_size": (64, 64),
        "block_stride": (32, 32),
        "cell_size": (32, 32),
        "bins": 8,
        "d_apperture": 1,
        "win_sigma": 4.,
        "hist_norm": 0,
        "l2_thresh": 0.2,
        "gamma_correction_factor": 0,
        "levels": 1
    }

    for sample in dataloader:
        out = compute_image_dct(image=sample[1])
        print(out)
        plt.imshow((out / 255), cmap="gray")
        plt.show()
        # hog_vector = compute_image_hog(sample[1], hog_params, mask=sample[2])

        # print(hog_vector.shape, hog_vector)
