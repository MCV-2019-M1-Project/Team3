import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Image Retrieval")
    parser.add_argument("--root_folder",
                        type=str,
                        default="data/dataset/",
                        help="Dataset root folder")

    # parametrization
    parser.add_argument(
        "--save", action="store_true", help="whether to save results"
    )

    parser.add_argument(
        "--dist",
        type=str,
        default="intersection",
        help="distance metric to use",
        choices=[
            "euclidean",
            "distance_L",
            "distance_x2",
            "intersection",
            "kl_divergence",
            "js_divergence",
            "hellinger",
        ],
    )

    parser.add_argument(
        "--color",
        type=str,
        default=None,
        help="color space to use",
        choices=["HSV", "LAB", "Gray", "XYZ", "HLS", "Luv", "YCbCr"],
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=64,
        help="number of bins to use in the histogram",
    )

    parser.add_argument(
        "--histogram",
        type=str,
        default="3D",
        help="type of histogram to use for FV",
        choices=["1D", "2D", "3D", "multiresolution", "pyramid"]
    )

    parser.add_argument(
        "--mr_splits",
        type=int,
        nargs=2,
        default=[1, 1],
        help="max number of splits on X and Y axis",
    )

    parser.add_argument(
        "--pyramid_rec_lvl",
        type=int,
        default=1,
        help=""
    )

    parser.add_argument(
        "--concat",
        action="store_true",
        help="whether to compute and concatenate a per-channel histogram",
    )

    parser.add_argument(
        "--sqrt",
        action="store_true",
        help="whether to apply the sqrt operation to the histogram",
    )

    # I/O
    parser.add_argument(
        "--output", type=str, default="output", help="path to save the results"
    )

    return parser.parse_args()
