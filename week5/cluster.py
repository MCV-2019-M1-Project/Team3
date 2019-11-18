import os
from tqdm import tqdm
from features import compute_hog, loc_bin_pat
from utils import group_paintings, mkdir, detect_denoise
from data.data import load_data
from opt import parse_args
import numpy as np
import cv2
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE


if __name__ == "__main__":

    args = parse_args()

    mkdir(args.output)
    log_path = os.path.join(args.output, "log.txt")

    images, query, gt, author_to_image = load_data(args)
    has_gt = bool(gt)

    # for bins in [2, 4, 8, 16]:
    for k in [5]:
        features = []
        np_images = []
        for i, img in enumerate(tqdm(images, total=len(images))):

            img = cv2.resize(img, (512, 512))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[..., 2]
            # np_images.append(img)

            # feature = cv2.calcHist(
            #     img, [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]
            # )
            feature, hog = compute_hog(img)
            feature = feature[:, None].astype("float32")
            np_images.append(hog)
            # feature = np.histogram(img, bins=bins)[0][..., None].astype("float32")
            # feature = cv2.normalize(feature, cv2.NORM_L2)
            features.append(feature.ravel())

        features = np.array(features)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.9)
        K = k
        ret, label, center = cv2.kmeans(
            features, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        # print(f"Bins {bins}, K {K}")
        print(f"Compactness {ret}")

        np_images = np.array(np_images)
        # tsne = TSNE(n_jobs=8)
        # y = tsne.fit_transform(features)
        # features = features.argsort()[-10:][::-1]

        for i in range(K):
            if i == 4:
                print(i)
                idx = label.squeeze() == i
                cluster = features[idx]
                hogs = np_images[idx][np.abs(cluster - center[i]).sum(1).argsort(0)]
                image_cluster = cluster.mean(0)
                # cv2.imshow(f"Type {i}", image_cluster)

                # cluster = np_images[features]
                # files = np.where(label.squeeze() == i)[0]
                files = np.abs(cluster - center[i]).sum(1).argsort(0)
                print(i)
                # plt.scatter(
                #     features[label.squeeze() == i, 0], features[label.squeeze() == i, 1], marker="."
                # )
                for j, img in enumerate(hogs[:20]):
                    cv2.imshow(f"hog", cv2.resize((img  * 255).astype("uint8"), (500, 500)))
                    cv2.imshow("corresp", cv2.resize(images[np.where(label.squeeze() == i)[0][files[j]]], (500, 500)))
                    print(np.where(label.squeeze() == i)[0][files[j]])
                    cv2.waitKey(0)
                    # plt.title(f"Cluster {i}, Image {j}")
                    # plt.show()
            # plt.show()
