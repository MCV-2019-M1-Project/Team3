import numpy as np
import cv2
import os

from distances import calculate_distances
from data.database import Database
from feature_extractor import FeatureExtractor
from utils import mask_background, estimate_background
from metrics import mapk


class Evaluator:
    def __init__(self, prototype_dict, output_folder):
        self.prototypes = prototype_dict
        self.feature_extractor = FeatureExtractor(None)
        self.feature_vector_protoypes = self.calc_FV_protoypes()
        self.output_folder = output_folder

    def calc_FV_protoypes(self, bins=1000, mask=None, sqrt=False):
        fvs = np.array([])
        for file, im in self.prototypes["images"].items():
            histogram = self.feature_extractor.compute_histogram(
                im, bins=bins, mask=mask, sqrt=sqrt
            )[..., None]
            fvs = np.append(fvs, histogram)
        return fvs.reshape(-1, 1000)

    def calc_FV_query(self, im, bins=1000, mask=None, sqrt=False):
        return self.feature_extractor.compute_histogram(
            im, bins=bins, mask=mask, sqrt=sqrt
        )

    def evaluate_query_set(self, query_set_dict, has_masks, distance_eq):
        gt = query_set_dict["gt"]
        predictions = []

        for file, im in query_set_dict["images"].items():

            mask = None
            if has_masks:
                mask_filename = os.path.join(
                    self.output_folder,
                    file.split("/")[-1].split(".")[0] + ".png",
                )
                mean_bgn = estimate_background(im, ratios=[0.1, 0.2, 0.3])
                im, mask = mask_background(im, mean_bgn)
                self.save_mask_file(mask_filename, mask)

            fv = self.calc_FV_query(im, mask=mask)
            distances = calculate_distances(
                self.feature_vector_protoypes, fv, distance_eq
            )

            predictions.append(list(distances.argsort()[:10]))

        map_k = mapk(gt, predictions)

        return map_k

    def save_mask_file(self, file_name, mask):
        if not cv2.imwrite(file_name, mask):
            raise Exception("Can't write to disk")

    def calc_mask_metrics(self, gt_mask, pred_mask):
        pass


if __name__ == "__main__":

    db = Database("data/dataset")
    evaluator = Evaluator(db.prototypes, "")

    dist_list = [
        "euclidean",
        "distance_L",
        "distance_x2",
        "intersection",
        "kl_divergence",
        "js_divergence",
        "hellinger",
    ]
    for dist in dist_list:
        print(dist)
        for query_set in db.query_sets:
            print(query_set["dataset_name"])
            has_masks = True if "masks" in query_set else False
            print(evaluator.evaluate_query_set(query_set, has_masks, dist))
