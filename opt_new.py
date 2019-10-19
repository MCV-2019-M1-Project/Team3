import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="Image Retrieval")
    parser.add_argument("--root_folder", type=str,
                        default="data/dataset/",
                        help="Dataset root folder")

    # parametrization
    parser.add_argument(
        "--save", action="store_true", help="whether to save results"
    )

    parser.add_argument(
        "--dist",
        type=str,
        default="kl_divergence",
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
        default=200,
        help="number of bins to use in the histogram",
    )

    parser.add_argument(
        "--type_histogram",
        type=str,
        default="1-D",
        help="type_histogram",
        choices=[
            "1-D",
            "2-D",
            "3-D",
            "multiblock-1-D",
            "multiblock-2-D",
            "multiblock-3-D",
        ],
    )

    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="level of block histogram",
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
