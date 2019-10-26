import os
from glob import glob

import cv2


class Dataloader:

    def __init__(self, path, compute_masks=False, detect_bboxes=False, evaluate=False):
        self.root = path
        self.image_files = sorted(glob(os.path.join(path, "*.jpg")))
        self.compute_masks = compute_masks
        self.detect_bboxes = detect_bboxes
        self.evaluate = evaluate
        if self.evaluate or self.compute_masks:
            self.masks_files = [f.replace("jpg", "png") for f in self.image_files]
        self.current = 0

    def load_image(self, im_file, mask_file):

        im = cv2.imread(im_file)

        mask = None
        if self.evaluate:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        return im, mask

    def __len__(self):
        return len(self.image_files)

    def __next__(self):
        try:
            mask = self.masks_files[self.current] if self.evaluate else None
            im, mask = self.load_image(self.image_files[self.current], mask)
            element = (self.image_files[self.current], im, mask)
        except IndexError:
            self.current = 0
            raise StopIteration
        self.current += 1
        return element

    def __iter__(self):
        return self


