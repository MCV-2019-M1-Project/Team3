import numpy as np
import cv2 as cv2
import glob
import utils
import random
import metrics
import pickle
from skimage.feature import (corner_peaks, corner_harris,
                             local_binary_pattern)

dataset = [cv2.imread(file) for file in glob.glob('E:\GitHub\Team3_2\data/bbdd\*.jpg')]
images = [cv2.imread(file) for file in glob.glob('E:\GitHub\Team3_2\data\qsd1_w4/*.jpg')]



f = open('E:\GitHub\Team3_2\data\qsd1_w4/gt_corresps.pkl','rb')
gt = pickle.load(f)
#%%

def keypoints(im):
    
    kp = corner_peaks(corner_shi_tomasi(im), min_distance=1)
    
    if np.shape(kp)[0]>=200:
        kp = kp[np.random.randint(0, np.shape(kp)[0]-1, 200)]

    else:
        add = 200-np.shape(kp)[0]
        rand_x = np.asarray(random.sample(range(np.shape(im)[0]), add))[...,None]
        rand_y = np.asarray(random.sample(range(np.shape(im)[1]), add))[...,None]
        
        rand = np.concatenate((rand_x,rand_y), axis = 1)
        kp = np.concatenate((kp, rand), axis = 0)
    return kp



    
def check_kp(im, kp):
    "Cut image to 16x16 at keypoint"
    p = 16
    im_rec = im[kp[0]-p:kp[0]+p, kp[1]-p:kp[1]+p]
    return im_rec   

  

def feature_descriptor(im_rec):
    """
    Args:
        im: grayscale image
        kp: key point, x and y coordinates of the grayscale image
    """
    hist = np.array([])
    a = np.array([10,20,30])    
    for i in range(0, np.shape(a)[0]):
        
        radius = a[i]
        n_points = 8
        METHOD = 'nri_uniform'
#        METHOD = 'ror'
        lbp = local_binary_pattern(im_rec, n_points, radius, METHOD)
        
        n_bins = 50
        hist =  np.concatenate((hist, np.histogram(lbp, n_bins, density = True)[0]))
       
    return hist

def keypoints2(im):
    im = im[16:-17, 16:-17]
    im = np.float32(im)
    harris = corner_harris(im)
    r = np.log(np.max((harris, np.ones((np.shape(harris)))*1000), axis = 0))
    kp = corner_peaks(r, min_distance=8)
    vals = r[kp[:,0],kp[:,1]]
    vals_arg = (-vals).argsort()[:200]
    kp = kp[vals_arg]
    
    if np.shape(kp)[0] < 200:
        add = 200-np.shape(kp)[0]
        p = 8
        rand_x = np.asarray(random.sample(range(p, np.shape(im)[0]-p), add))[...,None]
        rand_y = np.asarray(random.sample(range(p, np.shape(im)[1]-p), add))[...,None]
        
        rand = np.concatenate((rand_x,rand_y), axis = 1)
        kp = np.concatenate((kp, rand), axis = 0)
    
    kp = kp+16    
    return kp


#%%
    


lbp_hist_q = []
count = 0
for img in images:
    img, _, _, n = utils.detect_denoise(img, "best")
    multiple_painting, split_point, image_bg = utils.detect_paintings(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img*image_bg
    
    if multiple_painting == True:
        lbp_hist11 = []
        lbp_hist12 = []
        add = split_point-50
        img11 = utils.cut_image_gray(image_bg[:, :add], img[:, :add])
        img12 = utils.cut_image_gray(image_bg[:, add:], img[:, add:])
        
        img11 = np.float32(cv2.resize(img11, (500,500)))
        img12 = np.float32(cv2.resize(img12, (500,500)))

        kp1 = keypoints2(img11)
        for i in range(0,np.shape(kp1)[0]):
            im_rec1 = check_kp(img11, kp1[i])
            lbp_hist11.append(feature_descriptor(im_rec1))
        
        kp2 = keypoints2(img12)        
        for i in range(0,np.shape(kp2)[0]):
            im_rec2 = check_kp(img12, kp2[i])
            lbp_hist12.append(feature_descriptor(im_rec2))            
                
        lbp_hist_q.append([lbp_hist11, lbp_hist12])
                
    else:
        lbp_hist = []
        img = utils.cut_image_gray(image_bg, img)
        img = np.float32(cv2.resize(img, (500,500)))
        

        kp = keypoints2(img)
        for i in range(0,np.shape(kp)[0]):
            im_rec = check_kp(img, kp[i])
            lbp_hist.append(feature_descriptor(im_rec))   
                
        lbp_hist_q.append([lbp_hist])       
    
    print(count)
    count = count +1

#%%

lbp_hist_d = []
count = 0
for im in dataset:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.float32(cv2.resize(im, (500,500)))
    kp = keypoints2(im)
    lbp_hist = []
    for i in range(0,np.shape(kp)[0]):
        im_rec = check_kp(im, kp[i])
        lbp_hist.append(feature_descriptor(im_rec))

                
    lbp_hist_d.append(lbp_hist)       
    print(count)
    count = count +1


#%%

fqs = [val for p in lbp_hist_q for val in p]
fds = lbp_hist_d
            
            
def comp(f1,f2):
    'Distance comparation'
    n1, depth = np.shape(f1)  
    n2 = np.shape(f2)[0]
    d = np.zeros((n1,n2))
#    f1 = f1-np.mean(f1, axis = 1)[...,None]
#    f1 = f1/(np.std(f1)+0.00001)
#    f2 = f2-np.mean(f2, axis = 1)[...,None]
#    f2 = f2/(np.std(f2)+0.00001)
    for i in range(0, n1):
#        d[i, :] = np.sqrt(np.sum((np.tile(f1[i][None, ...], (n2,1))-f2)**2, axis = 1)) #L2
#        d[i, :] = np.sum(np.absolute(np.tile(f1[i][None, ...], (n2,1))-f2), axis = 1) #L1
        d[i,:] = 1/(0.00001+ np.sum(np.minimum(np.tile(f1[i][None, ...], (n2,1)), 
                                               f2), axis=1))
    ds = np.sort(d)           
    return ds  

def classif(ds, th, ratio):
    'Utility of matches classification with different criteria'
    ds0 = ds[:,0]
    ds1 = ds[:,1]
    r = ds0/(ds1+0.000001)
    
    cr1 = np.sum(ds0<th)
    cr2 = np.sum(r<ratio)
    cr3 = np.sum(ds0[ds0<th])
    
    if cr1!=0:
        cr3 = cr3/cr1
    else:
        cr3 = 0        
    cr4 = np.sum(ds0[ds0<th]**2)
    if cr1!=0:
        cr4 = cr4/cr1
    else:
        cr4 = 0
        
    cr5 = np.sum(ds0[r<ratio])
    if cr2!=0:
        cr5 = cr5/cr2
    else:
        cr5 = 0    
    cr6 = np.sum(ds0[r<ratio]**2)
    if cr2!=0:
        cr6 = cr6/cr2
    else:
        cr6 = 0 
        
    return np.array([cr1, cr2, cr3, cr4, cr5, cr6])

#a = comp(fqs[0], fds[162])
#crit = classif(a, 130, 0.7) 

nqs = np.shape(fqs)[0]
nds = np.shape(fds)[0]

crit = np.zeros((nqs, nds, 6))    
#a = comp(fqs[0], fds[35])
for qs in range(0,nqs):
    for ds in range(0,nds):
        dist = comp(fqs[qs], fds[ds])
        crit[qs,ds,:] = classif(dist, 0.3, 0.97)
#    print(qs)
#    print('all matches', np.argsort(-crit[qs,:,0])[:10])
#    print('all matches dist', -np.sort(-crit[qs,:,0])[:10])
#    print('select matches', np.argsort(-crit[qs,:,1])[:10])
#    print('select matches dist', -np.sort(-crit[qs,:,1])[:10])
    
#%%

preds2 = []
for i in range(0,39):
#    print(i, np.max(crit[i,:,1]))
    "Criteria to decide if that painting is on the dataset"
    
#    print(i)
#    print(np.argsort(-np.sqrt(crit[i,:,0]**2+1.0*crit[i,:,1]**2))[:10].tolist())
#    print((-np.sort(-np.sqrt(crit[i,:,0]**2+1.0*crit[i,:,1]**2))[:10]).tolist())
    if np.max(np.sqrt(crit[i,:,0]**2+1.0*crit[i,:,1]**2))<117:
        preds2.append(np.concatenate((np.array([-1]), 
                      np.argsort(-np.sqrt(crit[i,:,0]**2+1.7*crit[i,:,1]**2))[:9])).tolist())
    else:
        preds2.append(np.argsort(-np.sqrt(crit[i,:,0]**2+1.7*crit[i,:,1]**2))[:10].tolist())
      
for i in range(0,39):
    ini = np.argsort(-crit[qs,:,0])[:10]
    sel = np.argsort(-crit[qs,:,1])[:10]
    for j in range(0,10):
        if ini[j] in sel:
            match = ini[j]
            print(i,match)


gt_flat = [[val] for p in gt for val in p]
    
mapat1 = metrics.mapk(gt_flat, preds2, k=1)
print('Map@1: ', '{:.10f}'.format(mapat1))       
