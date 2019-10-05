import os
import cv2
import pickle


class Database:
    """ Class to load all datasets to work with,

    Args:
        root_path (str): path with datasets folders
        has_masks (bool): flag for searching gt masks

    Attributes:
        prototypes (dict): dictionary with prototype images data
        query_sets (list): list of dicrtionaries with validation and test sets.

    """

    def __init__(self, root_path, has_masks=False):

        self.prototypes = {}
        self.query_sets = []

        if os.path.exists(root_path):
            self.datasets = []

            for path in os.listdir(root_path):
                path = os.path.join(root_path, path)
                files = sorted(os.listdir(path))
                dataset_name = path.split("/")[-1]
                dataset_dict = {"dataset_name": dataset_name}
                if not files:
                    raise Exception("empty dataset")
                else:
                    images = [
                        os.path.join(path, x)
                        for x in files
                        if x.endswith(".jpg")
                    ]
                    dataset_dict["file_names"] = images
                    dataset_dict["images"] = self.load_dataset_images(images)

                if "qs" not in dataset_name:
                    self.prototypes = dataset_dict
                else:
                    if has_masks:
                        masks = [
                            os.path.join(path, x)
                            for x in files
                            if x.endswith(".png")
                        ]
                        dataset_dict["masks"] = self.load_dataset_images(masks)
                    gt = [
                        os.path.join(path, x)
                        for x in files
                        if x.endswith(".pkl")
                    ][0]
                    with open(gt, "rb") as f:
                        dataset_dict["gt"] = pickle.load(f)

                    self.query_sets.append(dataset_dict)
        else:
            raise Exception("root path not found")

    def load_image(self, filename):
        """
            Method that read an image from file and converts to RGB space
        Args:
            filename: image to be read

        Returns: Matrix with read image

        """
        im = cv2.imread(filename)

        if im is not None:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(" {}".format(filename))

    def load_dataset_images(self, filenames):
        """
        Function that loads all images and stores on a dictionary

        Args:
            filenames: list of images to read

        Returns: dictionary with images loaded (filename -> RGB matrix)

        """
        ims_dict = {}

        for im in filenames:
            ims_dict[im] = self.load_image(im)

        return ims_dict

    def __repr__(self):
        return "Database:\n \tPrototypes folder: {}\n \tQuery sets: {}".format(
            self.prototypes.keys(), [[*qs.keys()] for qs in self.query_sets]
        )


if __name__ == "__main__":

    db = Database("data")

    print(db.prototypes.keys())
