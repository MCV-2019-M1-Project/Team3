import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import transform_color, remove_background


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

    image = cv2.imread("data/dataset/query2/00001.jpg")[..., ::-1]
    image = transform_color(image, 'Gray')
    mask = cv2.imread("data/dataset/query2/00000.png")
    hist = f_extractor.compute_histogram(image)
    thr = hist.argmax()
    remove_background(image, thr)

