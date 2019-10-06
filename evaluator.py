import os

import numpy as np

from sklearn.metrics import recall_score, precision_score, f1_score

from distances import calculate_distances
from opt import parse_args
from data.database import Database
from feature_extractor import FeatureExtractor
from utils import mask_background
from utils import save_mask, save_predictions, mkdir
from metrics import mapk


class Evaluator:
    def __init__(self, prototype_dict, output_folder, opt):
        self.prototypes = prototype_dict
        self.opt = opt
        self.feature_extractor = FeatureExtractor(None)
        self.feature_vector_protoypes = self.calc_FV_protoypes()
        self.output_folder = output_folder
        self.metrics = {"precision": [], "recall": [], "f1": []}

    def calc_FV_protoypes(self, mask=None):
        fvs = np.array([])
        bins = self.opt.bins * 3 if self.opt.concat and self.opt.color != "Gray" else self.opt.bins
        for file, im in self.prototypes["images"].items():
            histogram = self.feature_extractor.compute_histogram(
                im,
                bins=self.opt.bins,
                mask=mask,
                sqrt=self.opt.sqrt,
                concat=self.opt.concat,
            )[..., None]
            fvs = np.append(fvs, histogram)

        return fvs.reshape(-1, bins)

    def calc_FV_query(self, im, mask=None):
        return self.feature_extractor.compute_histogram(
            im,
            bins=self.opt.bins,
            mask=mask,
            sqrt=self.opt.sqrt,
            concat=self.opt.concat,
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
                im, mask = mask_background(im)
                gt_mask = query_set_dict["masks"][file.replace("jpg", "png")]
                self.calc_mask_metrics(gt_mask[..., 0] / 255, mask / 255)
                if self.opt.save:
                    save_mask(mask_filename, mask)

            fv = self.calc_FV_query(im, mask)
            distances = calculate_distances(
                self.feature_vector_protoypes, fv, distance_eq
            )

            predictions.append(list(distances.argsort()[:10]))

        if self.opt.save:
            save_predictions(
                os.path.join(
                    self.output_folder,
                    "result_{}.pkl".format(int(has_masks) + 1),
                ),
                predictions,
            )

        map_k = mapk(gt, predictions)

        return map_k

    def calc_mask_metrics(self, gt_mask, pred_mask):

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

    db = Database(args.root_folder, has_masks=True, color_space=args.color)
    evaluator = Evaluator(db.prototypes, args.output, args)
    log = os.path.join(args.output, "log.txt")

    for query_set in db.query_sets:
        print(query_set["dataset_name"], file=open(log, "a"))
        print(query_set["dataset_name"])
        print(args.color)
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
        print({k:np.mean(v) for k, v in evaluator.metrics.items()})
