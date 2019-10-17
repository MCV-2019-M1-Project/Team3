from week2.feature_extractor import FeatureExtractor
from week2.dataloader import Dataloader
from week2.utils import remove_bg
from week2.distances import calculate_distances

import numpy as np

def calc_FV():
    pass


def calc_mapk():
    pass


def eval_masks():
    pass


if __name__ == '__main__':
    train = Dataloader("data/bbdd/")

    test_2_1 = Dataloader("data/qsd2_w1", has_masks=True)
    test_1_2 = Dataloader("data/qsd1_w2", has_masks=True)
    test_2_2 = Dataloader("data/qsd2_w2", has_masks=True)

    feature_extractor = FeatureExtractor()

    bbdd_FV = []
    for _, image, _ in train:
        bbdd_FV.append(feature_extractor.compute_histogram(image))

    bbdd_matrix = np.array(bbdd_FV)



    for _, image, _ in test_2_1:
        masks = remove_bg(image)
        #calc bboxes
        for mask in masks:
            query_fv = feature_extractor.compute_histogram(image,mask=mask)
            distances = calculate_distances(bbdd_matrix, query_fv)


            #eval mask
            #eval bbox

