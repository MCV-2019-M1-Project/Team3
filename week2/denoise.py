import os
import cv2
import numpy as np
from opt import parse_args
from utils import load_pickle, mkdir
from dataloader import Dataloader
from matplotlib import pyplot as plt
from feature_extractor import compute_histogram_1d
from skimage.restoration import estimate_sigma

def denoise_set(loader, opt):
    Noise_level_list=[]
    log_denoise = os.path.join(os.path.join(opt.output, loader.root.split("/")[-1]), "log_denoise.txt")
    log_file = open(log_denoise, "a")
    print(opt, file=log_file)
    for name, im, gt_mask in loader:
        #delect noise
        im, Noise_level_before, Noise_level_after, blur_type_last = detect_denoise(im, opt.blur_type)

        if opt.save_denoised_picture:
            save_pic(name.replace(test_1_3.root.split("/")[0], opt.output), im)

        print(name, Noise_level_before, Noise_level_after, blur_type_last, file=log_file)
    return

def detect_denoise(im, blur_type):
    #delect noise
    Noise_level_before=estimate_sigma(im,average_sigmas=True,multichannel=True)

    size_core = 3

    if Noise_level_before > 3.0:
        if blur_type == "GaussianBlur":
            blur_type_last = "GaussianBlur"
            im = cv2.GaussianBlur(im, (size_core, size_core), 0)
        elif blur_type == "medianBlur":
            blur_type_last = "medianBlur"
            im = cv2.medianBlur(im, size_core)
        elif blur_type == "blur":
            blur_type_last = "blur"
            im = cv2.blur(im,(size_core,size_core))
        elif blur_type == "bilateralFilter":
            blur_type_last = "bilateralFilter"
            im = cv2.bilateralFilter(im, size_core, 50, 50)
        elif blur_type == "best":
            Noise_level_after = 1000.0
            blur_type_last = "best"
            im_ori = im.copy()
            for blur_type_try in ["GaussianBlur", "medianBlur", "blur", "bilateralFilter"]:
                im_try = im_ori.copy()
                im_try,  Noise_level_before_try, Noise_level_after_try, blur_type_try2 = detect_denoise(im_try, blur_type_try)
                if (Noise_level_after_try<Noise_level_after):
                    im = im_try
                    Noise_level_after = Noise_level_after_try
                    blur_type_last = blur_type_try
        else:
            raise NotImplemented("you must choose from histograms types ")
    else:
        im = im
        blur_type_last = "none"

    Noise_level_after = estimate_sigma(im, average_sigmas=True, multichannel=True)

    return im, Noise_level_before, Noise_level_after, blur_type_last

def save_pic(file_name, pic):
    if not cv2.imwrite(file_name, pic):
        raise Exception("Can't write to disk")


if __name__ == '__main__':
    opt = parse_args()

    opt.histogram = "multiresolution"
    opt.color = "RGB"
    opt.bins = 256
    opt.concat = True
    opt.blur_type = "bilateralFilter"
    opt.save_denoised_picture = True

    os.chdir("..")
    mkdir(opt.output)
    log = os.path.join(opt.output, "log.txt")
    log_file = open(log, "a")
    print(opt, file=log_file)

    test_1_3 = Dataloader("data/qsd1_w3", evaluate=True)
    gt_1_3 = load_pickle("data/qsd1_w3/gt_corresps.pkl")
    tb_1_3 = load_pickle("data/qsd1_w3/text_boxes.pkl")
    mkdir(os.path.join(opt.output, test_1_3.root.split("/")[-1]))

    denoise_set(test_1_3, opt)

    '''
    path = "data/qsd1_w3"
    im_file = os.path.join(path, "00002.jpg")
    im = cv2.imread(im_file,0)

    hist = compute_histogram_1d(im)
    plt.plot(hist)
    cv2.imshow('im', im)




    hist = compute_histogram_1d(dst)
    plt.figure(2)
    plt.plot(hist)
    cv2.imshow('dst', dst)
    plt.show()



    cv2.imshow('im',im)
    dft = cv2.dft(np.float32(im), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # move to middle
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))   #painting
    plt.subplot(121), plt.imshow(im, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


    cv2.imshow('dst', dst)

    plt.figure(2)
    dft = cv2.dft(np.float32(dst), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # move to middle
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  # painting
    plt.subplot(121), plt.imshow(dst, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    pass
    
    '''