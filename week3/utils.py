import errno
import glob
import os
import pickle

import cv2
import numpy as np
from skimage import measure
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from scipy import ndimage


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_predictions(file_name, predictions):
    with open(file_name, "wb") as f:
        pickle.dump(predictions, f)


def save_mask(file_name, mask):
    if not cv2.imwrite(file_name, mask):
        raise Exception("Can't write to disk")


def transform_color(image, color_space):
    """Transforms an image bgr image to another color space

    Args:
        image (numpy.array): input rgb image
        color_space (string): desired color_space, e.g: HSV
    """

    color_transforms = {
        "RGB": cv2.COLOR_BGR2RGB,
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "YCbCr": cv2.COLOR_BGR2YCrCb,
        "Gray": cv2.COLOR_BGR2GRAY,
        "XYZ": cv2.COLOR_BGR2XYZ,
        "HLS": cv2.COLOR_BGR2HLS,
        "Luv": cv2.COLOR_BGR2Luv,
    }

    if color_space not in color_transforms:
        raise NotImplementedError(
            "Expected colorspace to be one of {}. Instead got {}".format(
                ", ".join([*color_transforms.keys()]), color_space
            )
        )

    return cv2.cvtColor(image, color_transforms[color_space])

def detect_paintings(img):
    """
    This function determines if there is 1 or 2 elements in one image.
    Args:
           img: image
    Returns:
           n_elements: number of elements in an image.
           arg: point at which we sould divide the image in the case there
           two elements.
    """
    sx, sy = np.shape(img)[:2]
    sx_mid = np.int(sx / 2)
    sy_mid = np.int(sy / 2)

    image_bg = remove_bg(img)
    lab = measure.label(image_bg)

    if np.max(lab) > 1:
        split_point = np.argmax(lab[sx_mid, :])
        if split_point < sy_mid:
            split_point = np.min(np.where(lab[sx_mid, :] == 1))
        else:
            split_point = split_point
    else:
        split_point = 0
    multiple_painting = np.max(lab) > 1
    return multiple_painting, split_point, image_bg


def detect_bboxes(im):
    """
    This function detects the text box of an image and calculates
    the mask to supress it.
    The function calls other functions that already caclulate the mask and
    coordinates for individual images.
    Args:
           images: set of images

    Returns:
           coord: list of list of lists of the coordinates of the text box.

           mask: list of binary images in the zone for the text box
    """

    multiple_paints, split_point, _ = detect_paintings(im)
    if multiple_paints:
        cut = np.int(split_point - 200)
        im1 = im[:, :cut, :]
        bbox1, submask_1 = detect_bbox(im1, 0)
        im2 = im[:, cut:, :]
        bbox2, submask_2 = detect_bbox(im2, cut)
        mask = np.concatenate((submask_1, submask_2), axis=1)
        return [bbox1, bbox2], mask
    else:
        bbox, mask = detect_bbox(im, 0)
        return [bbox], mask


def detect_bbox(image, add):
    """
    This function detects the text box of an image and calculates the mask
    to supress it
    USE THIS FUNCTION IF THERE IS ONLY ONE!! PAINTING PER IMAGE
    Args:
           image: image
           add: the x coordinate at which we had to cut an image with two
           paintings.
    Returns:
           coord: list of the coordinates of the text box.

           mask: binary image in the zone for the text box
    """
    sx, sy = np.shape(image)[:2]
    ker = np.ones((np.int(sx / 150), np.int(sy / 15)))
    ker2 = np.ones((15, 9))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mask1 = image[:, :, 1] == 128
    mask2 = image[:, :, 2] == 128

    d = np.uint8((mask1 * mask2) * 255)
    d = cv2.erode(d, ker, iterations=2)
    d = cv2.dilate(d, ker, iterations=2)
    lab = measure.label(d)

    if np.max(lab) > 1:
        assert (lab.max() != 0)  # assume at least 1 CC
        d = ((lab == np.argmax(np.bincount(lab.flat)[1:]) + 1) * 1).astype(np.uint8)
    else:
        d = d

    d = cv2.dilate(d, ker2, iterations=1)
    mask = d

    x, y, w, h = cv2.boundingRect(d)
    bbox = [x + add, y, x + w + add, y + h]

    return bbox, mask

def remove_bg(img):
    """
    This function removes the background from an input image
    Args:
           img: image
    Returns:
           filled: binary image of the background mask
    """
    sx, sy = np.shape(img)[:2]
    datatype = np.uint8

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(datatype)

    kernel2 = np.ones((90, 90))
    # We are going to use the saturation channel from HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]

    edges = cv2.Canny(img, 30, 30)

    # Closing to ensure edges are continuous
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Filling
    filled = (ndimage.binary_fill_holes(edges)).astype(np.float64)

    # Opening to remove imperfections
    filled = cv2.erode(filled, kernel2, iterations=1)
    filled = cv2.dilate(filled, kernel2, iterations=1)
    filled = filled.astype(np.uint8)

    # We remove the small elements which are not pictures
    lab = measure.label(filled)
    if np.max(lab) > 1:
        fi = np.zeros(np.shape(filled))
        for i in range(0, np.max(lab)):
            if np.sum((lab == (i + 1)) * 1) > 50000:
                a = lab == (i + 1)
                fi = a + fi
        filled = fi
    return filled



def remove_bg_improved(img, **kwargs):
    # grid search parameters
    kwargs = {
        "kernel": 5,
        "contrast": 1.3,
        "brightness": 9,
        "size": 1,
        "element": 0,
        "iters": 1,
        "sigma": 0.5,
    }

    contrast = kwargs["contrast"]
    brightness = kwargs["brightness"]
    iterations = kwargs["iters"]
    kernel = kwargs["kernel"]
    size = kwargs["size"]
    element = kwargs["element"]
    sigma = kwargs["sigma"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    adjusted = cv2.addWeighted(gray, contrast, gray, 0, brightness)

    # compute optimal canny upper and lower bounds
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(adjusted, lower, upper)
    element = cv2.getStructuringElement(
        element, (size * 2 + 1, size * 2 + 1), (size, size)
    )
    edges = cv2.dilate(edges, element)
    edges = cv2.erode(edges, element)
    contour_info = []
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    for c in contours:
        contour_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c)))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    mask = np.zeros(edges.shape)

    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))

    cc = measure.label(mask)
    masks = []
    if cc.max() > 1:
        fill = np.zeros(mask.shape)
        for i in range(1, cc.max() + 1):
            cc_i = cc == i
            if cc_i.sum() > 50000:
                fill = cc_i.astype('int32') + fill

        mask = fill

    masks.append(mask)
    for m in masks:
        m = cv2.dilate(m, None, iterations=iterations)
        m = cv2.erode(m, None, iterations=iterations)
        m = cv2.GaussianBlur(m, (kernel, kernel), 0)
        m = m.astype("uint8")
    return masks


def main():
    images = sorted(glob.glob("data/qsd2_w1/*.jpg"))
    masks = sorted(glob.glob("data/qsd2_w1/*.png"))
    metrics = {}

    preds = []
    gt = []

    args = {
        "kernel": 5,
        "contrast": 1.3,
        "brightness": 9,
        "size": 1,
        "element": 0,
        "iters": 1,
        "sigma": 0.5,
    }
    for path_img, path_mask in tqdm(zip(images, masks), total=len(images)):
        img = cv2.imread(path_img)
        gt_mask = cv2.imread(path_mask, 0)
        pred_mask = np.zeros_like(gt_mask, dtype=np.float64)

        for mask in remove_bg(img, **args):
            pred_mask += mask

        gt.append((cv2.resize(gt_mask, (500, 500)) / 255).astype("uint8"))
        preds.append(cv2.resize(pred_mask, (500, 500)))

        # cv2.imshow("org", cv2.resize(img, (500, 500)))
        # cv2.imshow("gt", cv2.resize(gt_mask, (500, 500)))
        # cv2.imshow("pred", preds[-1] * 255)
        # cv2.waitKey(0)

    metrics["precision"] = precision_score(
        np.array(gt).ravel(), np.array(preds).astype("uint8").ravel()
    )
    metrics["recall"] = recall_score(np.array(gt).ravel(), np.array(preds).astype("uint8").ravel())
    metrics["f1"] = f1_score(np.array(gt).ravel(), np.array(preds).astype("uint8").ravel())

    print(args)
    print(args, file=open("log.txt", "a"))
    print(metrics)
    print(metrics, file=open("log.txt", "a"))


def cut_image(mask, im):

    mask = mask * 1
    sx,sy= np.shape(mask)
    sx_mid = np.int(sx/2)
    sy_mid = np.int(sy/2)
    horiz = mask[sx_mid,:]
    verti = mask[:,sy_mid]
    h = np.where(horiz == 1)
    v = np.where(verti == 1)
    if h[0].size > 0:
        lx = np.min(h)
        rx = np.max(h)
    else:
        lx = 0
        rx = im.shape[1]
    if v[0].size > 0:
        ty = np.min(v)
        by = np.max(v)
    else:
        ty = 0
        by = im.shape[1]
    cut_im = im[ty:by, lx:rx, :]
#    print(lx,rx,ty,by)
    return cut_im

if __name__ == "__main__":
    main()
