import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

from week2.dataloader import Dataloader
from week2.distances import calculate_distances
from week2.feature_extractor import compute_histogram
from week2.metrics import mapk, averge_masks_metrics
from week2.opt import parse_args
from week2.utils import save_mask, load_pickle, save_predictions, detect_bboxes, detect_paintings


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
    for name, image, gt_mask in loader:
        multiple_painting, split_point, bg_mask = detect_paintings(image)
        bboxes, bbox_mask = detect_bboxes(image)
        res_mask = bg_mask - bbox_mask if loader.detect_bboxes else bg_mask
        if gt_mask is not None:
            calc_mask_metrics(masks_metrics, gt_mask / 255, bg_mask)
            save_mask(name.replace("/0", "/out/0").replace(".jpg", ".png"), res_mask * 255)


        # cropped sets, no need to mask image for retrieval
        if gt_mask is None:
            res_mask = None
        if loader.detect_bboxes:
            set_bboxes.append(bboxes)

        if multiple_painting:
            im_preds = []
            left_paint = np.zeros_like(res_mask)
            right_paint = np.zeros_like(res_mask)

            left_paint[:, split_point:] = res_mask[:, split_point:]
            left_paint[:, :split_point] = res_mask[:, :split_point]

            res_masks = [left_paint, right_paint]
            for submasks in res_masks:
                query_fv = calc_FV(image, opt, submasks).ravel()
                distances = calculate_distances(bbdd_fvs, query_fv, mode=opt.dist)
                im_preds.append((distances.argsort()[:10]).tolist())
            predictions.append(im_preds)

        else:
            query_fv = calc_FV(image, opt, res_mask).ravel()
            distances = calculate_distances(bbdd_fvs, query_fv, mode=opt.dist)

            predictions.append(list(distances.argsort()[:10]))

    save_predictions("results_{}.pkl".format(loader.root.split("/")[-1]), predictions)
    save_predictions("bboxes_{}.pkl".format(loader.root.split("/")[-1]), set_bboxes)

    map_k = [mapk(gt_correspondences, predictions, k=i) for i in [10, 3, 1]]
    avg_mask_metrics = averge_masks_metrics(masks_metrics)

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
    train = Dataloader("data/bbdd")

    test_2_1 = Dataloader("data/qsd2_w1", has_masks=True)
    gt_2_1 = load_pickle("data/qsd2_w1/gt_corresps.pkl")
    test_1_2 = Dataloader("data/qsd1_w2", detect_bboxes=True)
    gt_1_2 = load_pickle("data/qsd1_w2/gt_corresps.pkl")
    test_2_2 = Dataloader("data/qsd2_w2", has_masks=True, detect_bboxes=True)
    gt_2_2 = load_pickle("data/qsd2_w2/gt_corresps.pkl")

    bbdd_FV = []
    for _, image, _ in train:
        bbdd_FV.append(calc_FV(image, opt).ravel())

    bbdd_matrix = np.array(bbdd_FV)

    print(eval_set(test_2_1, gt_2_1, bbdd_matrix, opt))
    # print(eval_set(test_2_2, gt_2_2, bbdd_matrix, opt))
    print(eval_set(test_1_2, gt_1_2, bbdd_matrix, opt))
