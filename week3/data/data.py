import glob
import os
import pickle as pkl
import cv2

from tqdm import tqdm
from collections import OrderedDict
from utils import load_pickle

def load_data(args):

    data_path = os.path.join(args.root_folder, "bbdd")
    query_path = os.path.join(args.root_folder, args.query)
    text_path = os.path.join(args.root_folder, "bbdd_text")

    data_files = sorted(glob.glob(data_path + "/*.jpg"))
    query_files = sorted(glob.glob(query_path + "/*.jpg"))
    text_files = sorted(glob.glob(text_path + "/*.txt"))
    gt_file = sorted(glob.glob(query_path + "/*.pkl"))
    has_gt = bool(gt_file)

    author_to_image = get_authors(text_files)

    if has_gt:
        print("Loading Ground Truth")
        gt = load_pickle(gt_file[0])
    else:
        print("No Ground Truth Present")
        gt = []

    images = []
    query = []
    print("Loading BBDD Images")
    for path in tqdm(data_files, total=len(data_files)):
        images.append(cv2.imread(path))

    print("Loading Query Images")
    for path in tqdm(query_files, total=len(query_files)):
        query.append(cv2.imread(path))

    print("Loaded:")
    print("{} Database Images".format(len(images)))
    print("{} Query Images".format(len(query)))
    print("{} GT annotations".format(len(gt)))
    print("{} Authors File".format(len(author_to_image)))

    return images, query, gt, author_to_image


def get_authors(text_files):

    author_to_image = OrderedDict()
    for i, f in enumerate(text_files):
        text = open(f, "r").readlines()
        if text:
            author = text[0].split(",")[0].strip("('")
            author_to_image[i] = author
        else:
            author_to_image[i] = "Unknown"

    return author_to_image
