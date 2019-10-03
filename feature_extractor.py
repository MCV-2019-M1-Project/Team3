import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self):

        self.threshold = None

    def compute_histogram(self, image, mask=None):

        if mask is not None:
            mask = mask.astype("bool")

        hist = np.histogram(image[mask], bins=256, density=True)[0]

        return hist


if __name__ == "__main__":

    f_extractor = FeatureExtractor()

    image = np.random.rand(256, 256)
    mask = np.random.rand(256, 256)
    hist = f_extractor.compute_histogram(image, mask)
    print(hist.shape)
