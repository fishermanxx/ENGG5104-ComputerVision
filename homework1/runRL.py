import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy

def rlDeconv(B, PSF):
    # TODO Implement rl_deconv function based on spec.
    # Change based on your experiment
    maxIters = 25; 	

    pad_w = 30
    # Pad border to avoid artifacts
    I = np.pad(B, ((pad_w, pad_w), (pad_w, pad_w), (0,0)), 'edge')
    
    I0 = deepcopy(I)
    for i in range(0, maxIters):
    	# TODO 
        I = np.multiply(I, cv2.filter2D(np.divide(I0, cv2.filter2D(I, -1, PSF)), -1, cv2.flip(PSF, -1)))
    I = I[pad_w : -pad_w, pad_w : -pad_w]
    return I


if __name__ == '__main__':
    gt = cv2.imread('./misc/lena_gray.bmp').astype('double')
    gt = gt / 255.0

    print "gt's shape:" , gt.shape
    print "-----------------------------------------"

    
    # You can change to other PSF
    PSF = sio.loadmat('./misc/psf.mat')['PSF']
    
    print "PSF's shape:" , PSF.shape
    print "-----------------------------------------"

    
    # Generate blur image
    B = cv2.filter2D(gt, -1, PSF);
    
    print "B's shape:" , B.shape
    print "-----------------------------------------"

    
    # Show image, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('B',B)
    #   cv2.waitKey(0)
    # plt.imshow(B[:,:,[2,1,0]]) # for color image
    plt.imshow(B)
    plt.show()



    # Deconvolve image using RL
    I = rlDeconv(B, PSF)

    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('I',I)
    #   cv2.waitKey(0)
    plt.imshow(I)
    plt.show()

    
	
