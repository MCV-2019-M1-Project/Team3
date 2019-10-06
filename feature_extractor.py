import matplotlib.pyplot as plt
import numpy as np
from data.database import Database
from utils import mask_background, estimate_background


class FeatureExtractor:
    def __init__(self, dataset):
        """Class that will handle the feature extraction methods


        Args:
            dataset (Database): Database object containing the dataset
        """

        self.dataset = dataset

    @staticmethod
    def compute_histogram(img, bins=1000, mask=None, sqrt=False, concat=False):
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


if __name__ == "__main__":

    dataset = Database("data/dataset/", has_masks=True)
    f_extractor = FeatureExtractor(dataset)
    for img in dataset.query_sets[1]["images"].values():
        mean_bgn = estimate_background(img, ratios=[0.1, 0.2, 0.3, 0.4])
        img, mask = mask_background(img, mean_bgn)
        img_hist = f_extractor.compute_histogram(img, mask=(img != 0))
        plt.plot(img_hist)
        plt.show()
