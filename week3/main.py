import os
import warnings

from tqdm import tqdm
import numpy as np

from detect_text import retrieve_matches
from opt import parse_args
from data.data import load_data
from utils import detect_denoise, group_paintings, save_predictions, mkdir
from features import compare_ssim, loc_bin_pat, compute_image_dct, compute_mr_histogram
from metrics import mapk, sort
from distances import calculate_distances


def main():

    args = parse_args()

    mkdir(args.output)
    log_path = os.path.join(args.output, "log.txt")

    print(args.__dict__)
    print(args.__dict__, file=open(log_path, "a"))

    if "d2" in args.query or "t2" in args.query:
        process_bg = True
    else:
        process_bg = False

    if any([p not in ["lbp", "text", "color", "dct", "ssim"] for p in args.pipeline]):
        valid = '"lbp, "text", "color", "dct", "ssim"'
        raise ValueError(
            "Invalid option in pipeline. Expected any combination of {} but got {}".format(
                valid, args.pipeline
            )
        )

    # Load data
    images, query, gt, author_to_image = load_data(args)
    has_gt = bool(gt)

    preds = []
    authors = []

    print("Processing Query Set")
    for img in tqdm(query, total=len(query)):

        paintings = group_paintings(img, process_bg)
        im_preds = []

        for img in paintings:

            dists = []

            # Denoise
            img, _, _, _, = detect_denoise(img, blur_type="best")

            # Reduce search by matching authors
            if "text" in args.pipeline:
                matches, idxs, text = retrieve_matches(img, images, author_to_image)
                authors.append(text)

                im_preds.append(idxs.astype("int").tolist())  # if only text
            else:
                matches = images
                idxs = np.arange(len(matches))
                im_preds.append(0)  # will be replaced afterwards

            if "ssim" in args.pipeline:
                extractor = lambda x: x
                distance = None

            elif "lbp" in args.pipeline:
                extractor = loc_bin_pat
                distance = "hellinger"

            elif "dct" in args.pipeline:
                extractor = compute_image_dct
                distance = "euclidean"

            elif "color" in args.pipeline:
                extractor = compute_mr_histogram
                distance = "intersection"

            f2 = extractor(img)[None, ...]

            # Perform Retrieval
            for match in matches:

                if "ssim" in args.pipeline:
                    dist = compare_ssim(match, img.squeeze())

                else:
                    f1 = extractor(match)[None, ...]
                    dist = calculate_distances(f1, f2, mode=distance)[0]

                dists.append(dist)

            # replace if only text prediction
            im_preds[-1] = idxs[sort(dists)].astype("int").tolist()

        preds.append(im_preds)

    if has_gt:
        gt_flat = [[val] for p in gt for val in p]
        preds_flat = [val for p in preds for val in p]
        maps = [mapk(gt_flat, preds_flat, k=i) for i in [1, 3]]
        print("Map@{}: {}".format(1, maps[0]))
        print("Map@{}: {}".format(1, maps[0]), file=open(log_path, "a"))
        print("Map@{}: {}".format(3, maps[1]))
        print("Map@{}: {}".format(3, maps[1]), file=open(log_path, "a"))

    if args.save:

        save_predictions(
            os.path.join(args.output, "preds_{}.pkl".format(args.query)), preds
        )

        if "text" in args.pipeline:
            with open(
                os.path.join(args.output, "authors_{}.txt".format(args.query)), "w"
            ) as f:
                for author in authors:
                    f.write(author + "\n")


if __name__ == "__main__":

    main()
