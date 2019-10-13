from glob import glob
import cv2


class Dataloader:

    def __init__(self, path, has_masks=False):
        self.image_files = sorted(glob(path + "*.jpg"))
        self.has_masks = has_masks
        if self.has_masks:
            self.masks_files = sorted(glob(path + "*.png"))

    def load_image(self, im_file, mask_file):

        im = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

        mask = None
        if self.has_masks:
            mask = cv2.imread(mask_file)
        return im, mask

    def __getitem__(self, item):
        if item >= len(self.image_files):
            raise IndexError("trying to access item outside bounds")

        mask = self.masks_files[item] if self.has_masks else None
        im, mask = self.load_image(self.image_files[item], mask)

        return self.image_files[item], im, mask
