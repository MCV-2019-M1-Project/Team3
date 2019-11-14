import os
import warnings

from tqdm import tqdm
import numpy as np

from detect_text import retrieve_matches
from opt import parse_args
from data.data import load_data
from utils import detect_denoise, group_paintings, save_predictions, mkdir
from features import (
    compare_ssim,
    loc_bin_pat,
    loc_bin_pat_mr,
    compute_image_dct,
    compute_mr_histogram,
    surf_descriptor,
    compare_keypoints,
    filter_matches,
    calculate_match_dist,
    orb_descriptor, sift_descriptor,
)
from metrics import mapk, sort
from distances import calculate_distances


def main():

    args = parse_args()
    args.query = "qsd1_w4"
    args.pipeline = ["text", "lbp_mr"]
    os.chdir("..")

    dir = "E:\GitHub\Team3_2\My First Project-c10e01087b39.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = dir
    print(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))

    mkdir(args.output)
    log_path = os.path.join(args.output, "log.txt")

    print(args.__dict__)
    print(args.__dict__, file=open(log_path, "a"))


    process_bg = True

    if any(
        [
            p not in ["lbp", "lbp_mr", "text", "color", "dct", "ssim", "surf", "orb", "sift"]
            for p in args.pipeline
        ]
    ):
        valid = '"lbp, "lbp_mr", "text", "color", "dct", "ssim" "surf". "orb", "sift"'
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
    for i, img in enumerate(tqdm(query, total=len(query))):

        # Denoise
        img, _, _, _, = detect_denoise(img, blur_type="best")

        paintings = group_paintings(img, process_bg)
        im_preds = []

        for img in paintings:

            dists = []

            # Reduce search by matching authors
            if "text" in args.pipeline:
                matches, idxs, text, img = retrieve_matches(
                    img, images, author_to_image
                )
                authors.append(text)
                extractor = lambda x: x
                distance = None

                if matches:
                    im_preds.append(idxs.astype("int").tolist())  # if only text
            else:
                matches = images
                idxs = np.arange(len(matches))
                im_preds.append(0)  # will be replaced afterwards

            if not matches:
                im_preds.append([-1])

            else:
                if "ssim" in args.pipeline:
                    extractor = lambda x: x
                    distance = None

                elif "lbp" in args.pipeline:
                    extractor = loc_bin_pat
                    distance = "hellinger"

                elif "lbp_mr" in args.pipeline:
                    extractor = loc_bin_pat_mr
                    distance = "hellinger"

                elif "dct" in args.pipeline:
                    extractor = compute_image_dct
                    distance = "euclidean"

                elif "color" in args.pipeline:
                    extractor = compute_mr_histogram
                    distance = "intersection"

                elif "surf" in args.pipeline:
                    extractor = lambda x: x
                    distance = None

                f2 = extractor(img)[None, ...]

                # Perform Retrieval
                if len(args.pipeline) > 1 or "text" not in args.pipeline:
                    for match in matches:

                        if "ssim" in args.pipeline:
                            dist = compare_ssim(match, img.squeeze())

                        elif "surf" in args.pipeline:
                            image_key_points, image_descriptor = surf_descriptor(
                                img.squeeze()
                            )
                            match_key_points, match_descriptor = surf_descriptor(match)
                            im_matches = compare_keypoints(
                                match_descriptor, image_descriptor
                            )
                            im_matches = filter_matches(im_matches)
                            dist = calculate_match_dist(im_matches)

                        elif "orb" in args.pipeline:

                            image_key_points, image_descriptor = orb_descriptor(
                                img.squeeze()
                            )
                            match_key_points, match_descriptor = orb_descriptor(match)
                            im_matches = compare_keypoints(
                                match_descriptor, image_descriptor
                            )

                            im_matches = filter_matches(im_matches)
                            dist = calculate_match_dist(im_matches)

                        elif "sift" in args.pipeline:

                            image_key_points, image_descriptor = sift_descriptor(
                                img.squeeze()
                            )
                            match_key_points, match_descriptor = sift_descriptor(match)
                            im_matches = compare_keypoints(
                                match_descriptor, image_descriptor
                            )

                            im_matches = filter_matches(im_matches)
                            dist = calculate_match_dist(im_matches)

                        else:
                            f1 = extractor(match)[None, ...]
                            #dist = calculate_distances(f1, f2, mode="desc")
                            dist = calculate_distances(f1, f2, mode=distance)[0]

                        dists.append(dist)

                    # replace if only text prediction
                    im_preds[-1] = idxs[sort(dists)].astype("int").tolist()

                    if all(dist == np.inf for dist in dists):
                        if len(dists) != 1:
                            im_preds[-1] = [-1]

        preds.append(im_preds)

    if has_gt:
        gt_flat = [[val] for p in gt for val in p]
        preds_flat = [val for p in preds for val in p]
        maps = [mapk(gt_flat, preds_flat, k=i) for i in [1, 3, 5]]
        print("Map@{}: {}".format(1, maps[0]))
        print("Map@{}: {}".format(1, maps[0]), file=open(log_path, "a"))
        print("Map@{}: {}".format(3, maps[1]))
        print("Map@{}: {}".format(3, maps[1]), file=open(log_path, "a"))
        print("Map@{}: {}".format(5, maps[2]))
        print("Map@{}: {}".format(5, maps[2]), file=open(log_path, "a"))

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
