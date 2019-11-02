import cv2

def harris_laplacian(image):
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    harris_laplacian = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
    keypoints = harris_laplacian.detect(image)
    return keypoints

