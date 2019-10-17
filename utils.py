import cv2
import glob
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
from skimage import measure


def remove_bg(img, **kwargs):

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

    images = sorted(glob.glob("data/dataset/val2/*.jpg"))
    masks = sorted(glob.glob("data/dataset/val2/*.png"))
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
        preds.append(cv2.resize(remove_bg(img, **args), (500, 500)))
        mask = cv2.resize(cv2.imread(path_mask, 0), (500, 500))
        cv2.imshow("org", cv2.resize(img, (500, 500)))
        cv2.imshow("gt", mask)
        cv2.imshow("pred", preds[-1] * 255)
        cv2.waitKey(0)
        gt.append((mask / 255).astype("uint8"))

    metrics["precision"] = precision_score(
        np.array(gt).ravel(), np.array(preds).ravel()
    )
    metrics["recall"] = recall_score(np.array(gt).ravel(), np.array(preds).ravel())
    metrics["f1"] = f1_score(np.array(gt).ravel(), np.array(preds).ravel())

    print(args)
    print(args, file=open("log.txt", "a"))
    print(metrics)
    print(metrics, file=open("log.txt", "a"))


if __name__ == "__main__":
    main()
