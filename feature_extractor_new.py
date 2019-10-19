import matplotlib.pyplot as plt
import numpy as np
from data.database import Database
from utils import mask_background
import cv2
from opt_new import parse_args
from utils import mkdir


def compute_histogram(img, type_histogram="1-D", bins=1000, mask=None, sqrt=False, concat=False, level=1):
    if type_histogram == "1-D":
        return compute_1_d_histogram(img, bins, mask, sqrt, concat)
    elif type_histogram == "2-D":
        return compute_2_d_histogram(img, bins, mask, sqrt, concat)
    elif type_histogram == "3-D":
        return compute_3_d_histogram(img, bins, mask, sqrt, concat)
    elif (type_histogram == np.array(["multiblock-1-D", "multiblock-2-D", "multiblock-3-D"])).any():
        return compute_multiblock_histogram(img, bins, mask, sqrt, concat, level, type_histogram)
    else:
        raise Exception("wrong histogram option")

    pass


def compute_1_d_histogram(img, bins=1000, mask=None, sqrt=False, concat=False):
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

    HIST_SIZE = [bins]
    HIST_RANGE = [0, 256]
    CHANNELS = [0]

    if len(img.shape) == 3:
        if img.shape[2] == 3:
            histogram = np.array(
                [
                    cv2.calcHist([img], [i], mask, HIST_SIZE, HIST_RANGE)
                    for i in range(3)
                ]
            )
            histogram = histogram.ravel()
        else:
            raise Exception("Image should have more channels")
    else:
        histogram = cv2.calcHist([img], CHANNELS, mask, HIST_SIZE, HIST_RANGE)

    cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
    return histogram


def compute_2_d_histogram(img, bins=1000, mask=None, sqrt=False, concat=False, level=1):
    HIST_SIZE = [bins, bins]
    HIST_RANGE = [0, 256, 0, 256]
    CHANNELS = [0, 1]
    #CHANNELS = [1, 2]
    histogram = cv2.calcHist([img], CHANNELS, mask, HIST_SIZE, HIST_RANGE)
    cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
    return histogram


def compute_3_d_histogram(img, bins=1000, mask=None, sqrt=False, concat=False, level=1):
    HIST_SIZE = [bins, bins, bins]
    HIST_RANGE = [0, 256, 0, 256, 0, 256]
    CHANNELS = [0, 1, 2]
    histogram = cv2.calcHist([img], CHANNELS, mask, HIST_SIZE, HIST_RANGE)
    cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
    return histogram


def compute_multiblock_histogram(img, bins=1000, mask=None, sqrt=False, concat=False, level=1,
                                 type_histogram="multiblock-1-D"):
    """Computes the normalized density histogram of a given array

    Args:
        img (numpy.array): array of which to compute the histogram
        bins (int, optional): number of bins for histogram
        sqrt (bool, optional): whether to square root the computed histogram
        concat (bool, optional): whether to compute and concatenate per a
                                 channel histogram

        mask (numpy.array, optional): mask to apply to the input array
        level(int): divide the picture into 2**level * 2**level

    Returns:
        The computed histogram
    """

    # divide the image
    height = img.shape[0]
    width = img.shape[1]
    sub_number = 2 ** level
    sub_height = int(height / sub_number)
    sub_width = int(width / sub_number)
    subimage_list = []

    for i in range(0, sub_number):
        for j in range(0, sub_number):
            subimage = np.array(img[i * sub_height:(i + 1) * sub_height, j * sub_width:(j + 1) * sub_width, :])
            subimage_list.append(subimage)

    submask_list = None
    if mask is not None:
        submask_list = []
        for i in range(0, sub_number):
            for j in range(0, sub_number):
                submask = np.array(mask[i * sub_height:(i + 1) * sub_height, j * sub_width:(j + 1) * sub_width])
                submask_list.append(submask)

    subhist_list = []
    for i in range(0, sub_number * sub_number):
        subimage = subimage_list[i]
        if mask is not None:
            submask = submask_list[i]
        else:
            submask = None
        if type_histogram == "multiblock-1-D":
            subhist = compute_1_d_histogram(subimage, bins, submask, sqrt, concat)
        elif type_histogram == "multiblock-2-D":
            subhist = compute_2_d_histogram(subimage, bins, submask, sqrt, concat)
        elif type_histogram == "multiblock-3-D":
            subhist = compute_3_d_histogram(subimage, bins, submask, sqrt, concat)
        else:
            raise Exception("wrong histogram option")
        subhist_list = np.append(subhist_list, subhist)

    # return np.sqrt(subhist_list) if sqrt else subhist_list
    return subhist_list


if __name__ == "__main__":

    args = parse_args()
    state = args.__dict__
    print(state)

    mkdir(args.output)

    dataset = Database(args.root_folder, has_masks=True, color_space=args.color)

    for img in dataset.query_sets[1]["images"].values():
        img, mask = mask_background(img)
        hist = compute_histogram(img, mask=None, type_histogram="3-D", bins=200, level=1)
        """
        if args.type_histogram == "multiblock":
            # plot the result.
            fig1 = plt.figure(1)
            for i in range(1, sub_number * sub_number + 1):
                plt.subplot(sub_number, sub_number, i)
                plt.imshow(subimage_list[i - 1])
            fig2 = plt.figure(2)
            plt.imshow(mask)
            fig3 = plt.figure(3)
            for i in range(1, sub_number * sub_number + 1):
                plt.subplot(sub_number, sub_number, i)
                plt.plot(subhist_list[i - 1])
            plt.show()
          
            #plt.close(fig1)
            #plt.close(fig2)
            #plt.close(fig3)
            """

