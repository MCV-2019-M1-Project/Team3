import cv2
import glob
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract
from collections import OrderedDict
from fuzzywuzzy import fuzz


def detect_text(img):

    (W, H) = (320, 320)
    img = cv2.resize(img, (W, H))
    net = cv2.dnn.readNet("data/frozen_east_text_detection.pb")
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.7:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes
    results = []
    img_text = []
    for (startX, startY, endX, endY) in boxes:

        # extract the actual padded ROI
        roi = img[startY:endY, startX:endX]

        config = "-l eng --oem 1 --psm 11"

        text = pytesseract.image_to_string(roi, config=config)
        img_text.append(text)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))
        results = sorted(results, key=lambda r: r[0][1])

    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        dX = int((endX - startX) * 0.1)
        dY = int((endY - startY) * 0.1)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))

        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = img.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(
            output,
            text,
            (startX, startY - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    return img_text


def retrieve_match(text, database):
    pass



def main():

    author_to_image = OrderedDict()
    data = sorted(glob.glob("data/dataset/bbdd/*.jpg"))
    text_files = sorted(glob.glob("data/dataset/bbdd_text/*.txt"))
    for i, f in enumerate(text_files):
        text = open(f, "r").readlines()
        if text:
            author = text[0].split(",")[0].strip("('")
            author_to_image[i] = author
        else:
            author_to_image[i] = None

    query = sorted(glob.glob("data/dataset/qsd1_w3/*.jpg"))
    for img in query:
        text = detect_text(cv2.imread(img))
        text = " ".join(text)
        tokens = [
            fuzz.token_sort_ratio(text, author) if author is not None else 0
            for author in author_to_image.values()
        ]
        top10 = np.argpartition(np.array(tokens), -10)[-10:]

if __name__ == "__main__":

    main()
