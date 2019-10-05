import matplotlib.pyplot as plt
import numpy as np
from database import Database
from utils import mask_background, estimate_background


class FeatureExtractor:
    def __init__(self, dataset):
        """Class that will handle the feature extraction methods


        Args:
            dataset (Database): Database object containing the dataset
        """

        self.dataset = dataset

    @staticmethod
    def compute_histogram(img, mask=None):
        """Computes the normalized density histogram of a given array

        Args:
            img (numpy.array): array of which to compute the histogram
            mask (numpy.array, optional): mask to apply to the input array

        Returns:
            The computed histogram
        """

        if mask is not None:
            mask = mask.astype("bool")

        hist = np.histogram(img[mask], bins=1000, density=True)[0]

        # Add 1 to avoid division by 0 afterwards
        return hist + 1


if __name__ == "__main__":

    dataset = Database("data/dataset/", has_masks=True)
    f_extractor = FeatureExtractor(dataset)
    for img in dataset.query_sets[1]["images"].values():
        mean_bgn = estimate_background(img, ratios=[0.1, 0.2, 0.3, 0.4])
        img, mask = mask_background(img, mean_bgn)
        img_hist = f_extractor.compute_histogram(img, mask=(img != 0))
        plt.plot(img_hist)
        plt.show()
