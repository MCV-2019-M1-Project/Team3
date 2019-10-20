import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

from dataloader import Dataloader
from distances import calculate_distances
from feature_extractor import compute_histogram
from metrics import mapk, averge_masks_metrics
from opt import parse_args
from utils import save_mask, load_pickle, save_predictions, detect_bboxes, detect_paintings, mkdir, transform_color, \
    remove_bg
import os


def calc_FV(image, opt, mask=None):
    return compute_histogram(opt.histogram, image, splits=opt.mr_splits, rec_level=opt.pyramid_rec_lvl, bins=opt.bins,
                             mask=mask,
                             sqrt=opt.sqrt,
                             concat=opt.concat)


def eval_set(loader, gt_correspondences, bbdd_fvs, opt):
    masks_metrics = {"precision": [], "recall": [], "f1": []}
    ious = []
    predictions = []
    set_bboxes = []
    for name, query_image, gt_mask in loader:
        # transform to another color space
        multiple_painting, split_point, bg_mask = detect_paintings(query_image)
        bboxes, bbox_mask = detect_bboxes(query_image)
        res_mask = bg_mask.astype(bool) ^ bbox_mask.astype(bool) if loader.detect_bboxes else bg_mask
        if loader.compute_masks:
            if loader.evaluate:
                calc_mask_metrics(masks_metrics, gt_mask / 255, bg_mask)
            if opt.save:
                mask_name = name.split("/")[-1].replace(".jpg", ".png")
                save_mask(os.path.join(opt.output, loader.root.split("/")[-1], mask_name), res_mask * 255)

        # cropped sets, no need to mask image for retrieval
        if gt_mask is None:
            res_mask = None
        if loader.detect_bboxes:
            set_bboxes.append(bboxes)

        # change colorspace before computing feature vector
        query_image = transform_color(query_image, opt.color) if opt.color is not None else query_image
        if multiple_painting and gt_mask is not None:
            im_preds = []
            left_paint = np.zeros_like(res_mask)
            right_paint = np.zeros_like(res_mask)

            left_paint[:, split_point:] = res_mask[:, split_point:]
            right_paint[:, :split_point] = res_mask[:, :split_point]

            res_masks = [left_paint, right_paint]
            for submasks in res_masks:
                query_fv = calc_FV(query_image, opt, submasks).ravel()
                distances = calculate_distances(bbdd_fvs, query_fv, mode=opt.dist)
                im_preds.append((distances.argsort()[:10]).tolist())
            predictions.append(im_preds)

        else:
            query_fv = calc_FV(query_image, opt, res_mask).ravel()
            distances = calculate_distances(bbdd_fvs, query_fv, mode=opt.dist)

            predictions.append((distances.argsort()[:10]).tolist())

    if opt.save:
        save_predictions("{}/{}/result.pkl".format(opt.output, loader.root.split("/")[-1]), predictions)
        save_predictions("{}/{}/text_boxes.pkl".format(opt.output, loader.root.split("/")[-1]), set_bboxes)

    map_k = {i: mapk(gt_correspondences, predictions, k=i) for i in [10, 3, 1]} if loader.evaluate else None
    avg_mask_metrics = averge_masks_metrics(masks_metrics) if loader.evaluate else None

    return map_k, avg_mask_metrics


def calc_mask_metrics(out_dict, gt_mask, pred_mask):
    gt_mask = gt_mask.astype("uint8").ravel()
    pred_mask = pred_mask.astype("uint8").ravel()
    # TODO check what happen with some masks
    if pred_mask.max() != 1:
        pred_mask = (pred_mask / 255).astype("uint8")
    out_dict["recall"].append(recall_score(gt_mask, pred_mask))
    out_dict["precision"].append(precision_score(gt_mask, pred_mask))
    out_dict["f1"].append(f1_score(gt_mask, pred_mask))


if __name__ == '__main__':
    opt = parse_args()
    os.chdir("..")
    mkdir(opt.output)
    log = os.path.join(opt.output, "log.txt")
    log_file = open(log, "a")
    print(opt, file=log_file)

    train = Dataloader("data/bbdd")

    test_1_1 = Dataloader("data/qsd1_w1", evaluate=True)
    gt_1_1 = load_pickle("data/qsd1_w1/gt_corresps.pkl")
    mkdir(os.path.join(opt.output, test_1_1.root.split("/")[-1]))

    test_2_1 = Dataloader("data/qsd2_w1", compute_masks=True, evaluate=True)
    gt_2_1 = load_pickle("data/qsd2_w1/gt_corresps.pkl")
    mkdir(os.path.join(opt.output, test_2_1.root.split("/")[-1]))

    test_1_2 = Dataloader("data/qsd1_w2", detect_bboxes=True, evaluate=True)
    gt_1_2 = load_pickle("data/qsd1_w2/gt_corresps.pkl")
    mkdir(os.path.join(opt.output, test_1_2.root.split("/")[-1]))

    test_2_2 = Dataloader("data/qsd2_w2", compute_masks=True, detect_bboxes=True, evaluate=True)
    gt_2_2 = load_pickle("data/qsd2_w2/gt_corresps.pkl")
    mkdir(os.path.join(opt.output, test_2_2.root.split("/")[-1]))

    testset_1_2 = Dataloader("data/qst1_w2", detect_bboxes=True)
    mkdir(os.path.join(opt.output, testset_1_2.root.split("/")[-1]))

    testset_2_2 = Dataloader("data/qst2_w2", compute_masks=True, detect_bboxes=True)
    mkdir(os.path.join(opt.output, testset_2_2.root.split("/")[-1]))

    bbdd_matrix = np.array(
        [calc_FV(
            transform_color(image, opt.color) if opt.color is not None else image, opt).ravel()
                for _, image, _ in train
         ])
    print("Train sample loaded", bbdd_matrix.shape)

    print(test_1_1.root, file=log_file)
    print(eval_set(test_1_1, gt_1_1, bbdd_matrix, opt), file=log_file)

    print(test_2_1.root, file=log_file)
    print(eval_set(test_2_1, gt_2_1, bbdd_matrix, opt), file=log_file)

    print(test_1_2.root, file=log_file)
    print(eval_set(test_1_2, gt_1_2, bbdd_matrix, opt), file=log_file)

    print(test_2_2.root, file=log_file)
    print(eval_set(test_2_2, gt_2_2, bbdd_matrix, opt), file=log_file)

    print(testset_1_2.root, file=log_file)
    print(eval_set(testset_1_2, None, bbdd_matrix, opt), file=log_file)

    print(testset_2_2.root, file=log_file)
    print(eval_set(testset_2_2, None, bbdd_matrix, opt), file=log_file)