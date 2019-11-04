import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Image Retrieval")

    parser.add_argument(
        "--root_folder", type=str, default="data/", help="Dataset root folder"
    )

    parser.add_argument("--query", default="qsd1_w4", help="query set to evaluate")

    parser.add_argument("--save", action="store_true", help="whether to save results")

    parser.add_argument(
        "--pipeline",
        #default=["text", "ssim"],
        default=["ssim"],
        nargs='+',
        help="pipeline used to perform the retrieval"
    )
     # I/O
    parser.add_argument(
        "--output", type=str, default="output", help="path to save the results"
    )

    return parser.parse_args()
