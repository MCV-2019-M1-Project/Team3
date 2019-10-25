import numpy as np
import cv2
import glob
import pickle
import ssim_feature
import metrics
import denoise_ft
import matplotlib.pylab as plt
import utils


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
    
    
    
dist10 = []
dist3 = []
dist1 = []
distan = np.zeros((np.shape(dataset)[0]))
count2 = 0
for im in images_den:
    count = 0  
    for imd in dataset:
        distan[count] = ssim_feature.compare_ssim(im,imd)
        count = count+1
        
    dist10.append(((distan.argsort()[:10]).astype(int)).tolist())
    dist3.append(((distan.argsort()[:3]).astype(int)).tolist())
    dist1.append(((distan.argsort()[:1]).astype(int)).tolist())
    print(count2)    
    count2 = count2 + 1
    
mapat10 = metrics.mapk(gt, dist10, k=10)
mapat3 = metrics.mapk(gt, dist3, k=3)
mapat1 = metrics.mapk(gt, dist1, k=1)
print('Map@10: ', '{:.10f}'.format(mapat10),
      'Map@10: ', '{:.10f}'.format(mapat3), 
      'Map@1: ', '{:.10f}'.format(mapat1))





print('QSD 2')
dataset = [cv2.imread(file) for file in glob.glob('C:/Users/Sara/Datos/Master/M1/Project/dataset/*.jpg')]
images = [cv2.imread(file) for file in glob.glob('C:/Users/Sara/Datos/Master/M1/Project/week3/qsd2_w3/*.jpg')]


f = open('C:/Users/Sara/Datos/Master/M1/Project/week3/qsd2_w3/gt_corresps.pkl','rb')
gt = pickle.load(f)


dist10 = []
dist3 = []
dist1 = []
count2 = 0

for im in images:
    count = 0  
    multiple_painting, split_point, image_bg = utils.detect_paintings(im)
    distan1 = np.zeros((np.shape(dataset)[0]))
    distan2 = np.zeros((np.shape(dataset)[0]))
    if multiple_painting == True:
        add = split_point-100
        subim1 = ssim_feature.cut_image(image_bg[:,:add], im[:,:add,:])
        subim2 = ssim_feature.cut_image(image_bg[:,add:], im[:,add:,:])
        
        for imd in dataset:        
            distan1[count] = ssim_feature.compare_ssim(subim1,imd)
            distan2[count] = ssim_feature.compare_ssim(subim2,imd)
            count = count+1
            
        dist10.append([((distan1.argsort()[:10]).astype(int)).tolist(), 
                       ((distan2.argsort()[:10]).astype(int)).tolist()])
        
        dist3.append([((distan1.argsort()[:3]).astype(int)).tolist(), 
                       ((distan2.argsort()[:3]).astype(int)).tolist()])
        
        dist1.append([((distan1.argsort()[:1]).astype(int)).tolist(), 
                       ((distan2.argsort()[:1]).astype(int)).tolist()])            
        
    else:
        distan = np.zeros((np.shape(dataset)[0]))
        im = ssim_feature.cut_image(image_bg, im)
        for imd in dataset:  
            distan[count] = ssim_feature.compare_ssim(im,imd)
            count = count +1
            
        dist10.append(((distan.argsort()[:10]).astype(int)).tolist())
        dist3.append(((distan.argsort()[:3]).astype(int)).tolist())
        dist1.append(((distan.argsort()[:1]).astype(int)).tolist())
    print(count2)    
    count2 = count2 + 1
    
mapat10 = metrics.mapk(gt, dist10, k=10)
mapat3 = metrics.mapk(gt, dist3, k=3)
mapat1 = metrics.mapk(gt, dist1, k=1)
print('Map@10: ', '{:.10f}'.format(mapat10),
      'Map@10: ', '{:.10f}'.format(mapat3), 
      'Map@1: ', '{:.10f}'.format(mapat1))
