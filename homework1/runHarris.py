import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def harrisdetector(image, k, t):
    # TODO Write harrisdetector function based on the illustration in specification.
    # Return corner points x-coordinates in result[0] and y-coordinates in result[1]
    ptr_x = []
    ptr_y = []

    img = image.astype('double')  #(300L, 300L, 3L)
    Ix = cv2.filter2D(img,-1,np.array((-1,1)))
    Iy = cv2.filter2D(img,-1,np.array(((-1,1),)))
    w = np.ones((2*k+1,2*k+1))
    Ixx = cv2.filter2D(np.multiply(Ix, Ix), -1, w)  #(300L, 300L, 3L)
#    print ('Ixx shape: ',np.shape(Ixx))
    Ixy = cv2.filter2D(np.multiply(Ix, Iy), -1, w)
    Iyy = cv2.filter2D(np.multiply(Iy, Iy), -1, w)
    
    print 'starting to looking for the feature...\n'

    for i in range(k, np.size(image, 0) - k):
        for j in range(k, np.size(image, 1) - k):
            _, eig, _ = cv2.eigen(np.array(((Ixx[i,j,0], Ixy[i,j,0]), (Ixy[i,j,0], Iyy[i,j,0]))))
            if all(eig > t):
                ptr_x += [j]
                ptr_y += [i]
                
    result = [ptr_x, ptr_y]
    return result

if __name__ == '__main__':
    k = 2       # change to your value
    t = 23000     # change to your value

    I = cv2.imread('./misc/corner_gray.png')

    fr = harrisdetector(I, k, t)

    # Show input, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('output',out)
    #   cv2.waitKey(0)
    plt.imshow(I)
    # plot harris points overlaid on input image
    plt.scatter(x=fr[0], y=fr[1], c='r', s=40) 
    # show
    plt.show()							
