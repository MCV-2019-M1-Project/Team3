import matplotlib.pyplot as plt
import numpy as np
from data.database import Database
from utils import mask_background


def compute_multiblock_histogram(img, bins=1000, mask=None, sqrt=False, concat=False, level = 1):
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

    #crop the image
    height = img.shape[0]
    width = img.shape[1]
    sub_number = 2**level
    sub_height = int(height / sub_number)
    sub_width = int(width / sub_number)
    subimage_list = []

    for i in range(0, sub_number):
        for j in range(0, sub_number):
            subimage = np.array(img[i*sub_height:(i+1)*sub_height,j*sub_width:(j+1)*sub_width,:])
            subimage_list.append(subimage)


    if mask is not None:
        mask = mask.astype("bool")
        submask_list = []
        for i in range(0, sub_number):
            for j in range(0, sub_number):
                submask = np.array(mask[i * sub_height:(i + 1) * sub_height, j * sub_width:(j + 1) * sub_width, :])
                submask_list.append(submask)

    subhist_list = []
    for i in range(0, sub_number*sub_number):
        subimage = subimage_list[i]
        if len(subimage.shape) == 3:
            if concat:
                if subimage.shape[2] == 3:
                    submask = submask_list[i]
                    subhist = np.array(
                        [
                            np.histogram(
                                subimage[..., i][submask], bins=bins, density=True
                            )[0]
                            for i in range(3)
                        ]
                    )
                    subhist = subhist.ravel()
                else:
                    raise Exception("Image should have more channels")
            else:
                subhist = np.histogram(subimage[submask], bins=bins, density=True)[0]
        else:
            subhist = np.histogram(subimage[submask], bins=bins, density=True)[0]
        subhist_list.append(subhist)

    #return np.sqrt(subhist_list) if sqrt else subhist_list
    return subhist_list, subimage_list, submask_list, sub_number


if __name__ == "__main__":

    dataset = Database("data/dataset/", has_masks=True)
    for img in dataset.query_sets[1]["images"].values():
        img, mask = mask_background(img)
        subhist_list, subimage_list, submask_list, sub_number = compute_multiblock_histogram(img, mask=(img != 0), level=1)


        #plot the result.
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
        """
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        """




