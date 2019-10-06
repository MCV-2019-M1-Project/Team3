from distances import calculate_distances
from database import Database
from feature_extractor import FeatureExtractor
from utils import mask_background, estimate_background
from metrics import mapk, minimun_index_list
import numpy as np
import cv2
import os


class Evaluator:

    def __init__(self, prototype_dict, output_folder):
        self.prototypes = prototype_dict
        self.feature_extractor = FeatureExtractor(None)
        self.feature_vector_protoypes = self.calc_FV_protoypes()
        self.output_folder = output_folder

    def calc_FV_protoypes(self):
        fvs = np.array([])
        for file, im in self.prototypes['images'].items():
            histogram = self.feature_extractor.compute_histogram(im)[..., None]
            fvs = np.append(fvs, histogram)
        return fvs.reshape(-1, 1000)

    def calc_FV_query(self, im, mask=None):
        return self.feature_extractor.compute_histogram(im, mask=mask)

    def evaluate_query_set(self, query_set_dict, has_masks, distance_eq):
        gt = query_set_dict['gt']
        predictions = []

        for file, im in query_set_dict['images'].items():

            mask = None
            if has_masks:
                mask_filename = os.path.join(self.output_folder, file.split("/")[-1].split(".")[0] + ".png")
                mean_bgn = estimate_background(im, ratios=[0.1, 0.2, 0.3, 0.4])
                im, mask = mask_background(im, mean_bgn)
                self.save_mask_file(mask_filename, mask)

            fv = self.calc_FV_query(im, mask)
            distances = calculate_distances(self.feature_vector_protoypes, fv, distance_eq)

            predictions.append(minimun_index_list(distances))

<<<<<<< HEAD
        map = mapk(gt, predictions)

        return map
=======
        if self.opt.save:
            save_predictions(
                os.path.join(
                    self.output_folder,
                    "result_{}.pkl".format(int(has_masks) + 1),
                ),
                predictions,
            )

        map_k = mapk(gt, predictions)
>>>>>>> upstream/master

    def save_mask_file(self, file_name, mask):
        if not cv2.imwrite(file_name, mask):
            raise Exception("Can't write to disk")

    def calc_mask_metrics(self, gt_mask, pred_mask):
        pass


if __name__ == '__main__':

    db = Database("data")
    evaluator = Evaluator(db.prototypes, "")

<<<<<<< HEAD
    dist_list = ['euclidean', 'distance_L', 'distance_x2', 'intersection', 'kl_divergence',
                 'js_divergence', 'hellinger']
    for dist in dist_list:
        print(dist)
        for query_set in db.query_sets:
            print(query_set['dataset_name'])
            has_masks = True if 'masks' in query_set else False
            print(evaluator.evaluate_query_set(query_set, has_masks, dist))
=======
        gt_mask = gt_mask.ravel()
        pred_mask = pred_mask.ravel()
        self.metrics["recall"].append(recall_score(gt_mask, pred_mask))
        self.metrics["precision"].append(precision_score(gt_mask, pred_mask))
        self.metrics["f1"].append(f1_score(gt_mask, pred_mask))


if __name__ == "__main__":

    args = parse_args()
    state = args.__dict__
    print(state)

    mkdir(args.output)

    db = Database(args.root_folder, has_masks=True)
    evaluator = Evaluator(db.prototypes, args.output, args)
    log = os.path.join(args.output, "log.txt")

    for query_set in db.query_sets:
        print(query_set["dataset_name"], file=open(log, "a"))
        print(query_set["dataset_name"])
        has_masks = bool(query_set["masks"])
        print(
            "mapk: {:.4f}".format(
                evaluator.evaluate_query_set(query_set, has_masks, args.dist)
            )
        )
        print(
            "mapk: {:.4f}".format(
                evaluator.evaluate_query_set(query_set, has_masks, args.dist)
            ),
            file=open(log, "a"),
        )
>>>>>>> upstream/master
