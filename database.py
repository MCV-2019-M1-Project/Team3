import os
 
import cv2
 
 
class Database(object):
 
    def __init__(self, root_path, has_masks=False):
        """ TODO
 
        Args:
            root_path: path with datasets folders
            has_masks: flag for searching gt masks
        """
        if os.path.exists(root_path):
            self.datasets = []
 
            for path in os.listdir(root_path):
                path = os.path.join(root_path, path)
                files = sorted(os.listdir(path))
                dataset_name = path.split("/")[-1]
                dataset_dict = {
                    'dataset_name': dataset_name,
                    'is_proto': True if "qs" not in dataset_name else False}
                if not files:
                    raise Exception("empty dataset")
                else:
                    images = [os.path.join(path, x) for x in files if x.endswith(".jpg")]
                    dataset_dict['file_names'] = images
                    dataset_dict["images"] = self.load_dataset_images(images)
                    if has_masks:
                        masks = [os.path.join(path, x) for x in files if x.endswith(".png")]
                        dataset_dict["masks"] = self.load_dataset_images(masks)
 
                self.datasets.append(dataset_dict)
        else:
            raise Exception("root path not found")
 
    def load_image(self, filename):
        """
            Method that read an image from file and converts to RGB space
        Args:
            filename: image to be read
 
        Returns: Matrix with readed image
 
        """
        im = cv2.imread(filename)
 
        if im is not None:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            raise ("Error: File {} not found".format(filename))
 
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
 
    def __str__(self):
        return "Database: {}".format(self.datasets)
 
 
if __name__ == '__main__':
 
    db = Database("data")
 
    print(db)
 
    for dataset in db.datasets:
        print(dataset.keys())