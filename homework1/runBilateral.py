import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def gaussdis(x,y,temp):

    result = math.e**(-(x-y)**2 / temp**2)
    return result

def bilateral(image, sigma_s=5, sigma_r=0.1):
    # TODO Write "bilateral filter" function based on the illustration in specification.
    # Return filtered result image
    
    k = 3*sigma_s

    result = cv2.bilateralFilter(image, k, sigma_r, sigma_s)

    # test = []
    # result = image.copy()
    # for channel in range(3):
    #     im_now = result[:, :, channel]
    #     print "channel", channel, im_now.shape
    #     # for i in range(k, np.size(image, 0) - k):
    #     #     for j in range(k, np.size(image, 1) - k):
    #     #         print i, j
    #     #         devide = 0
    #     #         divisor = 0
    #     #         for m in range(-k, k):
    #     #             for n in range(-k, k):
    #     #                 gs = gaussdis(i+m, i, sigma_s)*gaussdis(j+n, j, sigma_s)
    #     #                 gr = gaussdis(im_now[i+m, j+n], im_now[i, j], sigma_r)
    #     #                 devide += gs*gr*im_now[i+m, j+n]
    #     #                 divisor += gs*gr
    #     #         result[i, j, channel] = np.divide(devide, divisor, dtype='float32')
    #     test.append(im_now)
    # test = [test[0], test[1], test[2]]
    # print test[0].shape
    # test = np.stack(test, axis=2)
    # print test.shape

    return 1 - result




if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp').astype(np.float32)
    print im.shape
    sigma_s = 5
    sigma_r = 0.1
    result = bilateral(im, sigma_s, sigma_r)
    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('output',result)
    #   cv2.waitKey(0)
    plt.imshow(result);
    plt.show()
