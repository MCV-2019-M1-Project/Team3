import cv2
import glob
import pickle
import lbp_feature
import distances
import numpy as np
import metrics
import utils
import denoise_ft


print('QSD-1')
dataset = [cv2.imread(file) for file in glob.glob('C:/Users/Sara/Datos/Master/M1/Project/dataset/*.jpg')]
images = [cv2.imread(file) for file in glob.glob('C:/Users/Sara/Datos/Master/M1/Project/week3/qsd1_w3/*.jpg')]


f = open('C:/Users/Sara/Datos/Master/M1/Project/week3/qsd1_w3/gt_corresps.pkl','rb')
gt = pickle.load(f)


images_den = []
count = 0
for im in images:
    im = denoise_ft.remove_noise_ft(images[count])
    images_den.append(im)
    count = count + 1
    print(count)


bins = 6
hist_im_query = np.zeros((np.shape(images)[0], bins))
count = 0
print('Test images')
for im in images_den:    
    lbp_im = lbp_feature.loc_bin_pat(im, bins)
    hist_im_query[count] = lbp_im
    count = count + 1 


hist_dataset = np.zeros((np.shape(dataset)[0], bins))
count = 0
print('Dataset images')
for im in dataset:    
    lbp_im = lbp_feature.loc_bin_pat(im, bins)
    hist_dataset[count] = lbp_im
    count = count +1
    
    

    
dist10 = []
dist3 = []
dist1 = []
for j in range(0, np.shape(images)[0]):
    dist_vector = distances.calculate_distances(hist_dataset, hist_im_query[j,:], mode='hellinger')    
    dist10.append(((dist_vector.argsort()[:10]).astype(int)).tolist())
    dist3.append(((dist_vector.argsort()[:3]).astype(int)).tolist())
    dist1.append(((dist_vector.argsort()[:1]).astype(int)).tolist())
    
    
    
mapat10 = metrics.mapk(gt, dist10, k=10)
mapat3 = metrics.mapk(gt, dist3, k=3)
mapat1 = metrics.mapk(gt, dist1, k=1)
print('Map@10: ', '{:.10f}'.format(mapat10),
      'Map@10: ', '{:.10f}'.format(mapat3), 
      'Map@1: ', '{:.10f}'.format(mapat1))




print('QSD-2')
dataset = [cv2.imread(file) for file in glob.glob('C:/Users/Sara/Datos/Master/M1/Project/dataset/*.jpg')]
images = [cv2.imread(file) for file in glob.glob('C:/Users/Sara/Datos/Master/M1/Project/week3/qsd2_w3/*.jpg')]


f = open('C:/Users/Sara/Datos/Master/M1/Project/week3/qsd2_w3/gt_corresps.pkl','rb')
gt = pickle.load(f)


images_den = []
count = 0
for im in images:
    im = denoise_ft.remove_noise_ft(images[count])
    images_den.append(im)
    count = count + 1
    print(count)

#%%
bins = 6

#hist_im_query = np.zeros((np.shape(images)[0], bins))
hist_im_query = []
count = 0
print('Test images')
for im in images_den:   
    multiple_painting, split_point, image_bg = utils.detect_paintings(images[count])
    if multiple_painting == True:
        print('Two paintings')
        add = split_point-100
        subim1 = lbp_feature.cut_image(image_bg[:,:add], im[:,:add,:])
        subim2 = lbp_feature.cut_image(image_bg[:,add:], im[:,add:,:])
        
        lbp_im1 = lbp_feature.loc_bin_pat(subim1, bins)
        lbp_im2 = lbp_feature.loc_bin_pat(subim2, bins)
        hist_im_query.append([lbp_im1, lbp_im2])
    else:
        lbp_im = lbp_feature.loc_bin_pat(lbp_feature.cut_image(image_bg, im), bins)
        hist_im_query.append(lbp_im)
    count = count+1
    

hist_dataset = np.zeros((np.shape(dataset)[0], bins))
count = 0
print('Dataset images')
for im in dataset:    
    lbp_im = lbp_feature.loc_bin_pat(im, bins)
    hist_dataset[count] = lbp_im
    count = count +1


dist10 = []
dist3 = []
dist1 = []
for j in range(0, np.shape(images)[0]):
    if np.shape(hist_im_query[j])[0] == 2:
        dist_vector1 = distances.calculate_distances(hist_dataset, hist_im_query[j][0], mode='hellinger')    
        dist_vector2 = distances.calculate_distances(hist_dataset, hist_im_query[j][1], mode='hellinger')    
        
        dist10.append([((dist_vector1.argsort()[:10]).astype(int)).tolist(), 
                       ((dist_vector2.argsort()[:10]).astype(int)).tolist()])
    
        dist3.append([((dist_vector1.argsort()[:3]).astype(int)).tolist(), 
                       ((dist_vector2.argsort()[:3]).astype(int)).tolist()])
    
        dist1.append([((dist_vector1.argsort()[:1]).astype(int)).tolist(), 
                       ((dist_vector2.argsort()[:1]).astype(int)).tolist()])
    
    
    else:
        dist_vector = distances.calculate_distances(hist_dataset, hist_im_query[j], mode='hellinger')    
        dist10.append(((dist_vector.argsort()[:10]).astype(int)).tolist())
        dist3.append(((dist_vector.argsort()[:3]).astype(int)).tolist())
        dist1.append(((dist_vector.argsort()[:1]).astype(int)).tolist())
    
    
    
mapat10 = metrics.mapk(gt, dist10, k=10)
mapat3 = metrics.mapk(gt, dist3, k=3)
mapat1 = metrics.mapk(gt, dist1, k=1)
print('Map@10: ', '{:.10f}'.format(mapat10),
      'Map@10: ', '{:.10f}'.format(mapat3), 
      'Map@1: ', '{:.10f}'.format(mapat1))

