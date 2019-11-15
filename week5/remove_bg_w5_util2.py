import errno
import os
import pickle

import cv2
import imutils
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
from skimage.restoration import estimate_sigma
from skimage.transform import hough_line, hough_line_peaks


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


def detect_denoise(im, blur_type):
    noise_level_before = estimate_sigma(im, average_sigmas=True, multichannel=True)

    if noise_level_before > 3.0:
        if blur_type == "GaussianBlur":
            blur_type_last = "GaussianBlur"
            im = cv2.GaussianBlur(im, (3, 3), 0)
        elif blur_type == "medianBlur":
            blur_type_last = "medianBlur"
            im = cv2.medianBlur(im, 3)
        elif blur_type == "blur":
            blur_type_last = "blur"
            im = cv2.blur(im, (3, 3))
        elif blur_type == "bilateralFilter":
            blur_type_last = "bilateralFilter"
            im = cv2.bilateralFilter(im, 7, 50, 50)
        elif blur_type == "best":
            noise_level_after = 1000.0
            blur_type_last = "best"
            im_ori = im.copy()
            for blur_type_try in [
                "GaussianBlur",
                "medianBlur",
                "blur",
                "bilateralFilter",
            ]:
                im_try = im_ori.copy()
                im_try, noise_level_before_try, noise_level_after_try, blur_type_try2 = detect_denoise(
                    im_try, blur_type_try
                )
                if noise_level_after_try < noise_level_after:
                    im = im_try
                    noise_level_after = noise_level_after_try
                    blur_type_last = blur_type_try
        else:
            raise NotImplementedError("you must choose from histograms types ")
    else:
        im = im
        blur_type_last = "none"

    noise_level_after = estimate_sigma(im, average_sigmas=True, multichannel=True)

    return im, noise_level_before, noise_level_after, blur_type_last


def group_paintings(img, process_bg):
    paintings = [img]
    if process_bg:
        group, split_point, mask = detect_paintings(img)
        if group:
            add = split_point - 100
            paintings = [
                cut_image(mask[:, :add], img[:, :add, :]),
                cut_image(mask[:, add:], img[:, add:, :]),
            ]

    return paintings


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


def detect_bboxes(im):
    """
    This function detects the text box of an image and calculates
    the mask to supress it.
    The function calls other functions that already caclulate the mask and
    coordinates for individual images.
    Args:
           im: image
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


def cut_image_gray(mask, im):
    mask = mask * 1
    sx, sy = np.shape(mask)
    sx_mid = np.int(sx / 2)
    sy_mid = np.int(sy / 2)
    horiz = mask[sx_mid, :]
    verti = mask[:, sy_mid]
    h = np.where(horiz == 1)
    v = np.where(verti == 1)
    lx = np.min(h)
    rx = np.max(h)
    ty = np.min(v)
    by = np.max(v)
    cut_im = im[ty:by, lx:rx]
    #    print(lx,rx,ty,by)
    return cut_im


def cut_image(mask, im):
    mask = mask * 1
    location_mask = np.where(mask == 1)
    sx_mid = np.int((location_mask[0].min() + location_mask[0].max()) / 2)
    sy_mid = np.int((location_mask[1].min() + location_mask[1].max()) / 2)

    horiz = mask[sx_mid, :]
    verti = mask[:, sy_mid]
    h = np.where(horiz == 1)
    v = np.where(verti == 1)
    lx = np.min(h)
    rx = np.max(h)
    ty = np.min(v)
    by = np.max(v)
    cut_im = im[ty:by, lx:rx, :]

    return cut_im


def find_lines(image):
    """
    Function that computes the hough transform of an image
    Args:
        image: grayscale image
    Returns:
        angs: array of the angles of the found lines
        distan: array of the distances of the found lines
    """
    image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = find_contours(image)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 540)
    h, theta, d = hough_line(edges, theta=tested_angles)
    angs = np.array([])
    distan = np.array([])
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=0.3 * np.max(h))):
        angs = np.append(angs, np.array([angle]))
        distan = np.append(distan, np.array([dist]))

    return angs, distan


def rotation(image, label, mask):
    """
    From a global image on the dataset it determines the rotation angle
    and rotates the image.
    Args:
        image: gray scale image
        label: labeled image of the mask
        mask: binary mask of where the paintings are
    Returns:
        rot_angle: shift of the angle. (we asume it is the same for all the paintings on the image)
        rotated_im: image rotated
        rotated_label = label rotated
        rotated_mask: mask rotated
    """
    image = np.uint8(image)
    mask = np.uint8(mask)

    angs, _ = find_lines(image)
    freqa = most_freq_angs(angs)

    if freqa[0] < -45:
        rot_angle = (180 - freqa[1]).astype(float)
        rotated_im = imutils.rotate(image, freqa[1])

        label = Image.fromarray(np.uint8(label))
        rotated_label = label.rotate(freqa[1], resample=Image.NEAREST)
        rotated_label = np.asarray(rotated_label)

        mask = Image.fromarray(np.uint8(mask))
        rotated_mask = mask.rotate(freqa[1], resample=Image.NEAREST)
        rotated_mask = np.asarray(rotated_mask)
        back_rot_angle = -freqa[1]
    else:
        rot_angle = np.abs(freqa[0]).astype(float)
        rotated_im = imutils.rotate(image, freqa[0])

        label = Image.fromarray(np.uint8(label))
        rotated_label = label.rotate(freqa[0], resample=Image.NEAREST)
        rotated_label = np.asarray(rotated_label)

        mask = Image.fromarray(np.uint8(mask))
        rotated_mask = mask.rotate(freqa[0], resample=Image.NEAREST)
        rotated_mask = np.asarray(rotated_mask)
        back_rot_angle = rot_angle

    return rot_angle, rotated_im, rotated_label, rotated_mask, back_rot_angle


def rotate_around_point_lowperf(point, radians, origin=(0, 0)):
    """Rotate a point around a given point.

    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + np.cos(radians) * (x - ox) + np.sin(radians) * (y - oy)
    qy = oy + -np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)

    return qx, qy


def cut_painting(image, args):
    """
    This function cuts the image into the different paintings
    Args:
        image: image to cut
        args: indices where the paintings end
    Returns:
        sub_images: list of images containing a single painting

    """
    sx, sy = np.shape(image)[:2]
    sub_images = []
    if sx > sy:
        for i in range(0, np.shape(args)[0]):
            if i == 0:
                sub_im = image[:args[i], :]
                sub_images.append(sub_im)
            elif i == np.shape(args)[0] - 1:
                sub_im = image[args[i - 1]:, :]
                sub_images.append(sub_im)
            else:
                sub_im = image[args[i - 1]:args[i], :]
                sub_images.append(sub_im)
    else:
        for i in range(0, np.shape(args)[0]):
            if i == 0:
                sub_im = image[:, :args[i]]
                sub_images.append(sub_im)
            elif i == np.shape(args)[0] - 1:
                sub_im = image[:, args[i - 1]:]
                sub_images.append(sub_im)

            else:
                sub_im = image[:, args[i - 1]:args[i]]
                sub_images.append(sub_im)

    return sub_images


def most_freq_angs(angs):
    """
    For a given array of detected angles it calculates the two most frequent
    It assumes that the most frequent value will be one of the ones that follow
    the sides of the painting
    Args:
        angs: Array containing all the angles found
    Returns:
        freqa: Two most frequent angles in degrees

    """
    hist_angs, bins = np.histogram(angs * 180 / np.pi, 540, [-90, 90], density=True)
    # arguments of the most frequent angles
    freq_angs = np.argsort(-hist_angs)
    freq_angs_sort = np.array([freq_angs[0]]).astype(int)
    # check if the two maximums are at least 85º apart and les than 95º
    # (optimal is 90º but it's a very hard restriction)
    for i in range(1, np.shape(freq_angs)[0] - 1):
        dif_angle = np.abs(bins[freq_angs[0] + 1] - bins[freq_angs[i] + 1])
        if dif_angle < 89.5 or dif_angle > 90.5:
            pass
        else:
            freq_angs_sort = np.append(freq_angs_sort, np.array([np.int(freq_angs[i] + 1)]))
            break

    freqa = np.array([bins[freq_angs_sort[0]], bins[freq_angs_sort[1]]])
    # ored from small to big the two most frequent angles
    freqa = np.sort(freqa)
    return freqa


def find_contours(image):
    """
    Canny edge detector
    Args:
        image: grayscale image
    Returns:
        edges: edges of the image
    """
    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(image, ret2 * 0.8, ret2)
    return edges


def mask_label(image):
    """
    Computes the mask and the labeled image of an input image. It resizes the image
    a certain factor depending on the size of  the image
    Args:
        image: grayscale image
    Return:
        filled: binary mask of the image
        lab: labeled image of the mask.
    """
    kernel = np.ones((5, 5))
    kernel2 = np.ones((90, 90))

    sx, sy = np.shape(image)[0:2]
    if sx > 2000 or sy > 2000:
        factor = 2
    else:
        factor = 1.5

    rx = np.int(sx / factor)
    ry = np.int(sy / factor)

    image = cv2.resize(image, (ry, rx))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    canny = find_contours(image)
    canny = cv2.dilate(canny, kernel, iterations=1)
    canny = cv2.erode(canny, kernel, iterations=1)
    _, cnts, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(np.shape(canny))
    for cnt in cnts:
        #        x,y,w,h = cv2.boundingRect(cnt)
        #        if w>150 and h>150:
        #            im = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        im = cv2.drawContours(mask, [cnt], 0, (0, 255, 0), 3)
    #    plt.figure()
    #    plt.imshow(im)
    im2 = np.ones(np.shape(canny)) - im
    im2[0:2, :] = 1
    im2[:, 0:2] = 1
    im2[-3:-1, :] = 1
    im2[:, -3:-3] = 1
    filled = (ndimage.binary_fill_holes(im2)).astype(np.float64)
    filled = cv2.erode(filled, kernel2, iterations=1)
    filled = cv2.dilate(filled, kernel2, iterations=1)
    filled = filled * 255
    filled = filled.astype(np.uint8)
    filled = cv2.resize(filled, (sy, sx), 0, 0, interpolation=cv2.INTER_NEAREST)
    lab = np.uint8(measure.label(filled))
    lab = cv2.resize(lab, (sy, sx), 0, 0, interpolation=cv2.INTER_NEAREST)
    return filled, lab


def detect_paintings(label):
    """
    Given a labeled image, detects the number of paintings
    It's assumed that the image is larger on the direction where the paintings
    are aranged
    Args:
        label: labeled image
    Returns:
        num_paintings: number of paintings in the image
        args: Arguments of the left or bottom side of each painting
    """
    args_r = np.array([]).astype(int)
    sx, sy = np.shape(label)[:2]
    sxm = np.int(sx / 2)
    sym = np.int(sy / 2)
    num_paintings = np.max(label)
    add = 10
    if num_paintings == 1:
        args_r = np.append(args_r, max(sx, sy))
        vertical = False
        return num_paintings, args_r, vertical
    else:

        for i in range(1, num_paintings + 1):
            if sx > sy:
                line_y = label[:, sym]
                arg_r = np.max(np.where(line_y == i)) + add
                args_r = np.append(args_r, arg_r)
                vertical = True
            else:
                line_x = label[sxm, :]
                arg_r = np.max(np.where(line_x == i)) + add
                args_r = np.append(args_r, arg_r)
                vertical = False
        args_r = np.sort(args_r)

        return num_paintings, args_r, vertical


def cut_painting_with_color(image, args):
    """
    This function cuts the image into the different paintings
    Args:
        image: image to cut
        args: indices where the paintings end
    Returns:
        sub_images: list of images containing a single painting
    """
    sx, sy = np.shape(image)[:2]
    sub_images = []
    if sx > sy:
        for i in range(0, np.shape(args)[0]):
            if i == 0:
                sub_im = image[:args[i], :, :]
                sub_images.append(sub_im)
            elif i == np.shape(args)[0] - 1:
                sub_im = image[args[i - 1]:, :, :]
                sub_images.append(sub_im)
            else:
                sub_im = image[args[i - 1]:args[i], :, :]
                sub_images.append(sub_im)
    else:
        for i in range(0, np.shape(args)[0]):
            if i == 0:
                sub_im = image[:, :args[i], :]
                sub_images.append(sub_im)
            elif i == np.shape(args)[0] - 1:
                sub_im = image[:, args[i - 1]:, :]
                sub_images.append(sub_im)

            else:
                sub_im = image[:, args[i - 1]:args[i], :]
                sub_images.append(sub_im)

    return sub_images


def useful_lines(image):
    """
    retruns only the lines at 0 and 90 º

    """
    sv, sh = np.shape(image)[0:2]
    print(sv, sh)
    angs, distan = find_lines(image)

    select_angs_h = np.array([])
    select_dist_h = np.array([])
    x_h = np.array([])

    select_angs_v = np.array([])
    select_dist_v = np.array([])
    y_v = np.array([])

    tol = 0.01
    for alph, dis in zip(angs, distan):
        if alph >= 0 - tol and alph <= 0 + tol:
            # vertical lines
            select_angs_v = np.append(select_angs_v, np.array([alph]))
            select_dist_v = np.append(select_dist_v, np.array([dis]))
            y_v = np.append(y_v, np.array([dis / np.sin(alph)]))
        if np.abs(alph) >= np.pi / 2 - tol and np.abs(alph) <= np.pi / 2 + tol:
            # Horizontal lines
            select_angs_h = np.append(select_angs_h, np.array([alph]))
            select_dist_h = np.append(select_dist_h, np.array([dis]))
            x_h = np.append(x_h, np.array([-dis / np.cos(alph)]))
        else:
            pass
    sort_d_h = np.argsort(np.abs(x_h))
    select_angs_h = select_angs_h[sort_d_h]
    select_dist_h = select_dist_h[sort_d_h]

    sort_d_v = np.argsort(y_v)
    select_angs_v = select_angs_v[sort_d_v]
    select_dist_v = select_dist_v[sort_d_v]

    return select_angs_h, select_dist_h, select_angs_v, select_dist_v


def corners_lines(angs_h, dist_h, angs_v, dist_v):
    # to cartesian coordinates
    p_corte = np.array([[0, 0]])
    for a_h, d_h in zip(angs_h, dist_h):
        for a_v, d_v in zip(angs_v, dist_v):
            a = np.array([[np.cos(a_v) / np.sin(a_v), 1], [np.cos(a_h) / np.sin(a_h), 1]])
            b = np.array([d_v / np.sin(a_v), d_h / np.sin(a_h)])
            x = np.uint(np.linalg.solve(a, b))
            p_corte = np.concatenate((p_corte, np.array([x])), axis=0)
    return p_corte[1:]


def mask_coordinates(mask, alpha, ox, oy, offset, vertical):
    """
    This function returns the coordinates of the masks
    Args:
        mask: the cutted mask containing a single painting rotated.
        alpha: the angle it has been rotated
        ox, oy: the center of the original image in respect to which the image
                was rotated
        offset: if the cutted mask corresponds to a second or third painting the
                number that needs to be added to the coordinates
        vertica: if the paintings are aligned vertically or horizontally
    return:
        rot_bounding_cords: the coordinates of a single painting on the image
    """
    alpha = alpha * np.pi / 180
    if np.max(mask) != 1:
        mask = (mask / np.max(mask)).astype(int)
    else:
        mask = mask.astype(int)

    sx, sy = np.shape(mask)
    location_mask = np.where(mask == 1)
    sxm = np.int((location_mask[0].min() + location_mask[0].max()) / 2)
    sym = np.int((location_mask[1].min() + location_mask[1].max()) / 2)

    line_x = mask[sxm, :]
    line_y = mask[:, sym]
    if vertical == True:
        r = np.max(np.where(line_x == 1))
        l = np.min(np.where(line_x == 1))
        t = np.min(np.where(line_y == 1)) + offset
        b = np.max(np.where(line_y == 1)) + offset
    else:
        r = np.max(np.where(line_x == 1)) + offset
        l = np.min(np.where(line_x == 1)) + offset
        t = np.min(np.where(line_y == 1))
        b = np.max(np.where(line_y == 1))

    bounding_cords = np.array([[l, t], [r, t], [r, b], [l, b]])
    rot_bounding_cords = np.zeros((4, 2))
    for i in range(0, 4):
        cord = bounding_cords[i]
        rot_cords = rotate_around_point_lowperf(cord, alpha, origin=(ox, oy))
        rot_bounding_cords[i, :] = rot_cords
    return bounding_cords, rot_bounding_cords


def group_paintings_rotation(img, process_bg):
    paintings = [img]

    if process_bg:
        im = img
        oy, ox = (np.shape(im)[:2])
        ox = np.int(ox / 2)
        oy = np.int(oy / 2)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask, label = mask_label(im)
        rot_angle, rotated_im, rotated_label, rotated_mask, back_angle_rot = rotation(im, label, mask)
        n_p, args, vertical = detect_paintings(rotated_label)

        rotated_im_with_color = imutils.rotate(img, -back_angle_rot)
        sub_ims = cut_painting_with_color(rotated_im_with_color, args)
        sub_mask = cut_painting(rotated_mask / 255, args)
        paintings = [cut_image(sub_mask[i], sub_ims[i]) for i in range(0, len(args))]
    return paintings


def list_ang_cord(im):
    lista = []
    oy, ox = (np.shape(im)[:2])
    ox = np.int(ox / 2)
    oy = np.int(oy / 2)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask, label = mask_label(im)

    rot_angle, rotated_im, rotated_label, rotated_mask, back_angle_rot = rotation(im, label, mask)
    n_p, args, vertical = detect_paintings(rotated_label)
    sub_mask = cut_painting(rotated_mask, args)

    if n_p == 1:
        lista_1 = []
        bounding_cords, rot_bounding_cords = mask_coordinates(rotated_mask, back_angle_rot, ox, oy, 0, vertical)
        lista_1.append(rot_angle)
        lista_1.append(bounding_cords.tolist())
        lista.append(lista_1)

    else:
        for i in range(0, n_p):
            if i == 0:
                arg = 0
                lista_1 = []
            else:
                arg = args[i - 1]
                lista_1 = []
            bounding_cords, rot_bounding_cords = mask_coordinates(sub_mask[i], back_angle_rot, ox, oy, arg, vertical)

            lista_1.append(rot_angle)
            lista_1.append(bounding_cords.tolist())
            lista.append(lista_1)

    return lista
