import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
from skimage.transform import hough_line, hough_line_peaks
import glob
import matplotlib.pylab as plt
from matplotlib import cm
import utils
import pickle
import imutils
import imageio
from sklearn import metrics
from skimage.feature import corner_peaks, corner_harris, peak_local_max, corner_fast
from skimage.filters import sobel
from PIL import Image
import PIL




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
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold = 0.3*np.max(h))):
        angs = np.append(angs, np.array([angle]))
        distan = np.append(distan, np.array([dist]))
        
    return angs, distan




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
    hist_angs, bins = np.histogram(angs*180/np.pi, 540, [-90,90], density=True)
    #arguments of the most frequent angles
    freq_angs = np.argsort(-hist_angs)
    freq_angs_sort = np.array([freq_angs[0]]).astype(int)
    #check if the two maximums are at least 85º apart and les than 95º
    #(optimal is 90º but it's a very hard restriction)
    for i in range(1,np.shape(freq_angs)[0]-1):
        dif_angle = np.abs(bins[freq_angs[0]+1]-bins[freq_angs[i]+1])
        if dif_angle<89.5 or dif_angle>90.5:
            pass
        else:
            freq_angs_sort = np.append(freq_angs_sort, np.array([np.int(freq_angs[i]+1)]))
            break
        
    freqa = np.array([bins[freq_angs_sort[0]],bins[freq_angs_sort[1]]])
    #ored from small to big the two most frequent angles
    freqa = np.sort(freqa)
    return freqa




def find_contours(image):
    """
    Canny edge detector
    Args:
        Image: grayscale image
    Returns:
        edges: edges of the image
    """
    ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(image, ret2*0.8, ret2)
    return edges




def mask_label(image):
    """
    Computes the mask and the labeled image of an input image. It resizes the image
    a certain factor depending on the size of  the image
    Args:
        Image: grayscale image
    Return: 
        filled: binary mask of the image
        lab: labeled image of the mask. 
    """
    kernel = np.ones((5,5))
    kernel2 = np.ones((90,90))

    sx,sy = np.shape(image)[0:2]
    if sx>2000 or sy>2000:
        factor = 2
    else:
        factor = 1.5    
    
    rx = np.int(sx/factor)
    ry = np.int(sy/factor)

    image = cv2.resize(image, (ry,rx))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    canny = find_contours(image)
    canny =  cv2.dilate(canny,kernel,iterations = 1)
    canny =  cv2.erode(canny,kernel,iterations = 1)
    cnts,_ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(np.shape(canny))
    for cnt in cnts:
#        x,y,w,h = cv2.boundingRect(cnt)
#        if w>150 and h>150:
#            im = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        im = cv2.drawContours(mask,[cnt], 0, (0,255,0), 3)
#    plt.figure()
#    plt.imshow(im)
    im2 = np.ones(np.shape(canny)) - im
    im2[0:2,:] = 1
    im2[:,0:2] = 1
    im2[-3:-1 , :] = 1
    im2[:, -3:-3] = 1
    filled = (ndimage.binary_fill_holes(im2)).astype(np.float64)
    filled = cv2.erode(filled, kernel2, iterations = 1)
    filled = cv2.dilate(filled, kernel2, iterations = 1)
    filled = filled * 255
    filled = filled.astype(np.uint8)
    filled = cv2.resize(filled, (sy,sx), 0,0, interpolation = cv2.INTER_NEAREST)
    lab = np.uint8(measure.label(filled))
    lab = cv2.resize(lab, (sy,sx), 0,0, interpolation = cv2.INTER_NEAREST)
    return filled, lab




def rotation(image, label,  mask):
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

    if freqa[0]<-45:
        rot_angle = (180-freqa[1]).astype(float)
        rotated_im = imutils.rotate(image, freqa[1])
        
        label = Image.fromarray(np.uint8(label))
        rotated_label = label.rotate( freqa[1], resample = PIL.Image.NEAREST)
        rotated_label = np.asarray(rotated_label)
        
        mask = Image.fromarray(np.uint8(mask))
        rotated_mask = mask.rotate( freqa[1], resample = PIL.Image.NEAREST)
        rotated_mask = np.asarray(rotated_mask)
        back_rot_angle = -freqa[1]
    else:
        rot_angle = np.abs(freqa[0]).astype(float)
        rotated_im = imutils.rotate(image, -rot_angle)
        
        label = Image.fromarray(np.uint8(label))
        rotated_label = label.rotate( -rot_angle, resample = PIL.Image.NEAREST)
        rotated_label = np.asarray(rotated_label)
        
        mask = Image.fromarray(np.uint8(mask))
        rotated_mask = mask.rotate( -rot_angle, resample = PIL.Image.NEAREST)
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
    sxm = np.int(sx/2)
    sym = np.int(sy/2)
    num_paintings = np.max(label)
    add = 10
    if num_paintings==1:          
        args_r = np.append(args_r,0)
        vertical = False
        return num_paintings, args_r, vertical
    else:
       
        for i in range(1, num_paintings+1):
            if sx>sy:
                line_y = label[:, sym]
                arg_r = np.max(np.where(line_y==i)) + add
                args_r = np.append(args_r, arg_r)
                vertical = True
            else:
                line_x = label[sxm, :]
                arg_r = np.max(np.where(line_x==i)) + add
                args_r = np.append(args_r, arg_r)
                vertical = False
        args_r = np.sort(args_r)
        
        return num_paintings, args_r, vertical
    


def cut_painting(image, args):
    """
    This function cuts the image into the different paintings
    Args:
        image: image to cut
        args: indices where the paintings end
    Returns:
        sub_images: list of images containing a single painting
        
    """
    sx,sy = np.shape(image)[:2]
    sub_images = []
    if sx>sy:
        for i in range(0, np.shape(args)[0]):    
            if i==0:
                sub_im = image[:args[i],:]
                sub_images.append(sub_im)
            elif i == np.shape(args)[0]-1:
                sub_im = image[args[i-1]:,:]
                sub_images.append(sub_im)
            else:
                sub_im = image[args[i-1]:args[i], :]
                sub_images.append(sub_im)
    else:
        for i in range(0, np.shape(args)[0]):    
            if i==0:
                sub_im = image[:, :args[i]]
                sub_images.append(sub_im)
            elif i ==np.shape(args)[0]-1:
                sub_im = image[:, args[i-1]:]
                sub_images.append(sub_im)
                
            else:
                sub_im = image[:, args[i-1]:args[i]]
                sub_images.append(sub_im)
                
    return sub_images



def pres_rec_f1_2(GT, pred):

       recall = np.zeros(30)
       precision = np.zeros(30)
       f1 = np.zeros(30)
       count = 0
       for groundt in GT:
              sx,sy = np.shape(groundt)[:2]
              max_val = np.max(pred[count])
              if max_val == 0:
                  max_val = 1
              recall[count] = metrics.recall_score(np.reshape(groundt[:,:,0], sx*sy)/255, np.reshape(pred[count], sx*sy)/max_val)
              precision[count] = metrics.precision_score(np.reshape(groundt[:,:,0], sx*sy)/255, np.reshape(pred[count], sx*sy)/max_val)
              f1[count] = metrics.f1_score(np.reshape(groundt[:,:,0], sx*sy)/255, np.reshape(pred[count], sx*sy)/max_val)
              count = count +1
       return np.mean(precision), np.mean(recall), np.mean(f1)
   

    
def useful_lines(image):
    """
    retruns only the lines at 0 and 90 º
    
    """
    sv, sh = np.shape(image)[0:2]
    print(sv, sh)
    angs, distan = find_lines(image)
       
    select_angs_h = np.array([])
    select_dist_h = np.array([])
    x_h =  np.array([])
    
    select_angs_v = np.array([])
    select_dist_v = np.array([])
    y_v = np.array([])
    
    tol = 0.01
    for alph, dis in zip(angs, distan):
        if alph>=0-tol and alph<=0+tol:
            #vertical lines
            select_angs_v = np.append(select_angs_v, np.array([alph]))   
            select_dist_v = np.append(select_dist_v, np.array([dis]))   
            y_v = np.append(y_v, np.array([dis/np.sin(alph)]))
        if np.abs(alph)>=np.pi/2-tol and np.abs(alph)<=np.pi/2+tol:
            #Horizontal lines
            select_angs_h = np.append(select_angs_h, np.array([alph]))        
            select_dist_h = np.append(select_dist_h, np.array([dis]))   
            x_h = np.append(x_h, np.array([-dis/np.cos(alph)]))
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
    #to cartesian coordinates
    p_corte = np.array([[0,0]])
    for a_h, d_h in zip(angs_h, dist_h):
        for a_v, d_v in zip(angs_v, dist_v):
            a = np.array([[np.cos(a_v)/np.sin(a_v),1],[np.cos(a_h)/np.sin(a_h),1]])
            b = np.array([d_v/np.sin(a_v), d_h/np.sin(a_h)])
            x = np.uint(np.linalg.solve(a, b))
            p_corte = np.concatenate((p_corte, np.array([x])), axis = 0)
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
    alpha = alpha*np.pi/180
    if np.max(mask) != 1:
        mask = (mask/np.max(mask)).astype(int)
    else:
        mask = mask.astype(int)
        
    sy, sx = np.shape(mask)
    sym = np.int(sy/2)
    sxm = np.int(sx/2)
    
    line_x = mask[sym, :]
    line_y = mask[:, sxm]
    if vertical == True:
        r = np.max(np.where(line_x==1))
        l = np.min(np.where(line_x==1))
        t = np.min(np.where(line_y==1)) + offset
        b = np.max(np.where(line_y==1)) + offset
    else:
        r = np.max(np.where(line_x==1)) + offset
        l = np.min(np.where(line_x==1)) + offset
        t = np.min(np.where(line_y==1)) 
        b = np.max(np.where(line_y==1)) 
    
    bounding_cords = np.array([[l,t], [r,t], [r,b], [l,b]])
    rot_bounding_cords = np.zeros((4,2))
    for i in range(0,4):
        cord = bounding_cords[i]
        rot_cords = rotate_around_point_lowperf(cord, alpha, origin=(ox, oy))
        rot_bounding_cords[i,:] = rot_cords
    return bounding_cords, rot_bounding_cords

    #%%
images = [cv2.imread(file) for file in glob.glob("C:/Users/Sara/Datos/Master/M1/Project/week5/qsd1_w5/*.jpg")]
GTS = [cv2.imread(file) for file in glob.glob("C:/Users/Sara/Datos/Master/M1/Project/week5/qsd1_w5/*.png")]

#%%
#example on an image
im = images[10]
oy, ox = (np.shape(im)[:2])
ox = np.int(ox/2)
oy = np.int(oy/2)

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
mask, label = mask_label(im)
rot_angle, rotated_im, rotated_label, rotated_mask, back_angle_rot = rotation(im, label, mask)
n_p, args, vertical = detect_paintings(rotated_label)
sub_ims = cut_painting(rotated_im, args)
sub_mask = cut_painting(rotated_mask, args)

plt.imshow(im, cmap = 'gray')
if n_p == 1:
    bounding_cords, rot_bounding_cords = mask_coordinates(rotated_mask, back_angle_rot, ox,oy, 0, vertical)
    plt.plot(rot_bounding_cords[:,0], rot_bounding_cords[:,1], 'o')
    
else:
    for i in range(0,n_p):
        if i == 0:
            arg = 0
        else:
            arg = args[i-1]
        bounding_cords, rot_bounding_cords = mask_coordinates(sub_mask[i], back_angle_rot, ox,oy, arg, vertical)
        plt.plot(rot_bounding_cords[:,0], rot_bounding_cords[:,1], 'o')
#%%

c = 0
masks = []
for im in images[0:1]:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask, label = mask_label(im)
    masks.append(mask)
#    imageio.imsave('mask{}.png'.format(c), mask)
    rot_angle, rotated_im, rotated_label, rotated_mask = rotation(im, label, mask)

    c = c +1

#p,r,f = pres_rec_f1_2(GTS, masks)         
#print('Precision: ', p)   
#print('Precision: ', r)
#print('Precision: ', f)
