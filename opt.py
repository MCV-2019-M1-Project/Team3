import argparse

def parse_args():

    # Required
    parser = argparse.ArgumentParser(description="Simple Image Retrieval")
    parser.add_argument("root", type=str, help="Dataset root folder")

    # Optimization
    parser.add_argument("--lr", "-l", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size.")
    parser.add_argument("--schedule", type=int, nargs='*', default=[])
    parser.add_argument("--margin", type=float, default=0.3, help="margin for loss")
    parser.add_argument("--test_only", action="store_true", help="Perform test step alone.")
    parser.add_argument("--val_set", default="val1", help="validation set to use")
    parser.add_argument("--k", type=int, default=1, help="k for map")

    # I/O
    parser.add_argument("--output", type=str, default="output", help="output path")
    parser.add_argument("--size", type=int, default=256, help="original image size")

    # Performance
    parser.add_argument("--num_workers", type=int, default=8, help="Number of prefetching threads")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of gpus to use")

    return parser.parse_args()
