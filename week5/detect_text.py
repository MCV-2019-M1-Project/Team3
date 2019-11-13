import glob
import numpy as np
from google.cloud import vision
import io
import cv2
from tqdm import tqdm
#from similarity.normalized_levenshtein import NormalizedLevenshtein
from strsimpy import NormalizedLevenshtein


def retrieve_matches(img, database, author_to_image):

    normalized_levenshtein = NormalizedLevenshtein()
    text, boxes = detect_text(img)
    if text == None:
        return [], [], [], img
    sims = []
    for word in text:
        sims.append(
            np.array(
                [
                    normalized_levenshtein.similarity(word, author)
                    for author in author_to_image.values()
                ]
            )
        )
    ssims = sorted(sims, key=lambda x: x.sum())
    argmax = sorted(range(len(sims)), key=lambda x: sims[x].sum())[-1]
    idxs = np.where(ssims[-1] > 0.8)[0]
    author = text[argmax]
    box = boxes[argmax] + boxes[argmax + len(author.split()) - 1]
    coords_x = sorted(box, key=lambda x: x[0])
    coords_y = sorted(box, key=lambda x: x[1])
    coords = [coords_y[0][1], coords_y[-1][1], coords_x[0][0], coords_x[-1][0]]
    offset_x = 0#int(abs(coords[2] - coords[3]))
    offset_y = 0#int(abs(coords[0] - coords[1]))
    mask = np.ones(img.shape[:2])
    mask[coords[0] - offset_y:coords[1]+offset_y, coords[2]-offset_x:coords[3]+offset_x] = 0


    # if no high confidence match was found get the 10 highest
    if idxs.size == 0:
        # idxs = ssims[-1].argsort()[-10:][::-1]
        return [], [], author, img

    matches = []
    for match in idxs:
        matches.append(database[match])

    return matches, idxs, author, (img * mask[..., None]).astype("uint8")


def detect_text(img):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    image = vision.types.Image(content=cv2.imencode(".jpg", img)[1].tostring())

    response = client.text_detection(image=image)
    texts = response.text_annotations

    vertices = []
    for text in texts:

        vertices.append(([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]))

    if len(texts) == 0:
        text = None
    else:
        text = texts[0].description.split("\n")

    return text, vertices


def main():

    data = sorted(glob.glob("data/dataset/bbdd/*.jpg"))
    text_files = sorted(glob.glob("data/dataset/bbdd_text/*.txt"))
    query = sorted(glob.glob("data/dataset/qsd1_w3/*.jpg"))
    gt = sorted(glob.glob("data/dataset/qsd1_w3/*.pkl"))
    has_gt = bool(gt)

    images = []
    print("Loading Images")
    for path in tqdm(data, total=len(data)):
        images.append(cv2.imread(path))

    author_to_image = get_authors(text_files)

    print("Processing Query Set")
    for img in tqdm(query, total=len(query)):
        matches = retrieve_matches(img, images, author_to_image)


if __name__ == "__main__":

    main()
