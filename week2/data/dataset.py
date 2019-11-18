import glob
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MuseumDatasetTest(Dataset):

    def __init__(self, path, transforms, val_set):

        if val_set == "val1":
            self.labels = pickle.load(open(path + "val1/gt_corresps.pkl", "rb"))
            self.images = sorted(glob.glob(path + "val1/*.jpg"))

        else:
            self.labels = pickle.load(open(path + "val2/gt_corresps.pkl", "rb"))
            self.images = sorted(glob.glob(path + "val2/*.jpg"))

        self.p_images = sorted(glob.glob(path + "train/*.jpg"))
        self.transforms = transforms
        self.label_mat = np.zeros((len(self.images), len(self.p_images)))
        for i, label in enumerate(self.labels):
            for idx in label:
                self.label_mat[i, idx] = 1

        self.pairs = self._build_pairs_()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = Image.open(self.images[idx]).convert("RGB")

        img = self.transforms(img)

        return img

    def _build_pairs_(self):

        gpairs = list(np.where((self.label_mat == 1)))
        gpairs.append(np.ones(gpairs[0].size))
        pairs = gpairs

        return pairs


class MuseumDataset(Dataset):
    def __init__(self, path, transforms, train):

        labels1 = pickle.load(open(path + "val1/gt_corresps.pkl", "rb"))
        labels2 = pickle.load(open(path + "val2/gt_corresps.pkl", "rb"))

        if train:
            self.images = sorted(glob.glob(path + "val1/*.jpg"))
            self.images += sorted(glob.glob(path + "val2/*.jpg"))[:20]
            self.labels = labels1 + labels2[:20]

        else:
            self.images = sorted(glob.glob(path + "val2/*.jpg"))[20:]
            self.labels = labels2[20:]

        self.p_images = sorted(glob.glob(path + "train/*.jpg"))
        self.transforms = transforms
        self.label_mat = np.zeros((len(self.images), len(self.p_images)))
        for i, label in enumerate(self.labels):
            for idx in label:
                self.label_mat[i, idx] = 1

        self.pairs = self._build_pairs_()

    def __len__(self):
        return len(self.pairs[0])

    def __getitem__(self, idx):

        label = self.pairs[2][idx]
        idx1 = self.pairs[0][idx]
        idx2 = self.pairs[1][idx]
        img1 = Image.open(self.images[idx1]).convert("RGB")
        img2 = Image.open(self.p_images[idx2]).convert("RGB")

        img1, img2 = self.transforms(img1), self.transforms(img2)


        return (img1, img2), label, (idx1, idx2)

    def _build_pairs_(self):

        gpairs = list(np.where((self.label_mat == 1)))
        ipairs = list(np.where(self.label_mat == 0))

        ipairs.append(np.zeros(ipairs[0].size) - 1)
        gpairs.append(np.ones(gpairs[0].size))
        pairs = [np.concatenate((g, i), 0) for g, i in zip(gpairs, ipairs)]

        return pairs
