import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.sparse.linalg as sslg
import scipy.sparse as ss
import flowTools as ft


def estimateHSflowlayer(frame1, frame2, uv, lam=80, maxwarping=1):
    H, W = frame1.shape
    print 'H: ', H, 'W: ', W

    npixels = H * W
    print 'npixels: ', npixels
    
    x, y = np.meshgrid(range(W), range(H))

    # TODO#3: build differential matrix and Laplacian matrix according to
    # image size
    e = np.ones([npixels])
    Dy = ss.spdiags([-e, e], [0, 1], npixels, npixels, format = 'csc' )  #Dx?
    Dx = ss.spdiags([-e, e], [0, H], npixels, npixels, format = 'csc' )  #Dy?
    L = Dx.T.dot(Dx) + Dy.T.dot(Dy)

    # please use order 'F' when doing reshape: np.reshape(xx, xxshape, order='F') 

    # Kernel to get gradient
    h = np.array( [[1, -8, 0, 8, -1]], dtype='single' ) / 12

    for i in range(maxwarping):

        # TODO#2: warp image using the flow vector
        # an example is in runFlow.py
        remap = np.zeros([H, W, 2])
        remap[:, :, 0] = x + uv[:, :, 0]
        remap[:, :, 1] = y + uv[:, :, 1]
        remap = remap.astype('single')
        warped2 = cv2.remap(frame2, remap, None, cv2.INTER_CUBIC)      

        # TODO#4: compute image gradient Ix, Iy, and Iz
        Ix = cv2.filter2D( warped2, -1, h )
        Iy = cv2.filter2D( warped2, -1, h.transpose() )
        Iz = warped2 - frame1

        # TODO#5: build linear system to solve HS flow
        # generate A,b for linear equation Ax = b
        # you may need use scipy.sparse.spdiags

        U = uv[:, :, 0].reshape((npixels, 1), order='F')
        V = uv[:, :, 1].reshape((npixels, 1), order='F')
        Ix = np.reshape(Ix, (npixels, 1), order='F')
        Iy = np.reshape(Iy, (npixels, 1), order='F')
        Iz = np.reshape(Iz, (npixels, 1), order='F')
     
        Ix = ss.spdiags(Ix.T, [0], npixels, npixels, format = 'csc' )
        Iy = ss.spdiags(Iy.T, [0], npixels, npixels, format = 'csc' )

        A11 = Ix*Ix + lam*L
#        print 'A11 shape: ', A11.shape
        A12 = Ix*Iy
#        print 'A12 shape: ', A12.shape
        A21 = Ix*Iy
#        print 'A21 shape: ', A21.shape
        A22 = Iy*Iy + lam*L
#        print 'A22 shape: ', A22.shape
        A = ss.vstack( [ss.hstack([A11,A12]), ss.hstack([A21,A22])] )
        print 'A shape: ', A.shape

        b1 = Ix*Iz + lam*L*U
        b2 = Iy*Iz + lam*L*V
        b = -np.vstack( [b1, b2] )
    
        ret = sslg.spsolve(A, b)
        deltauv = np.reshape(ret, uv.shape, order='F')

        deltauv[deltauv is np.nan] = 0
        deltauv[deltauv > 1] = 1
        deltauv[deltauv < -1] = -1

        uv = uv + deltauv

#        print 'Warping step: %d, Incremental norm: %3.5f' %(i, np.linalg.norm(deltauv)
        # Output flow
    return uv


def estimateHSflow(frame1, frame2, lam = 80):
    H, W = frame1.shape
    print 'H: ',H, 'W: ', W
    # build the image pyramid
    pyramid_spacing = 2.0

    ##16*(2^level)=min(W, H)
    pyramid_levels = 1 + np.floor(np.log(min(W, H) / 16.0) / np.log(pyramid_spacing * 1.0))
    pyramid_levels = int(pyramid_levels)
    smooth_sigma = np.sqrt(2.0)

    pyramid1 = []
    pyramid2 = []

    pyramid1.append(frame1)
    pyramid2.append(frame2)

    for m in range(1, pyramid_levels):
        # TODO #1: build Gaussian pyramid for coarse-to-fine optical flow
        # estimation
        # use cv2.GaussianBlur
        temp1 = cv2.GaussianBlur(pyramid1[m-1], (5,5), smooth_sigma)
        temp2 = cv2.GaussianBlur(pyramid2[m-1], (5,5), smooth_sigma)
        
        H1_new = int(temp1.shape[0]*0.5)
        W1_new = int(temp1.shape[1]*0.5)
        H2_new = int(temp2.shape[0]*0.5)
        W2_new = int(temp2.shape[1]*0.5)
        
        temp3 = cv2.resize(temp1, (W1_new, H1_new))
        pyramid1.append(temp3)
        temp4 = cv2.resize(temp2, (W2_new, H2_new))
        pyramid2.append(temp4)
    # coarst-to-fine compute the flow
    uv = np.zeros(((H, W, 2)))

    for levels in range(pyramid_levels - 1, -1, -1):
        print "level %d" % (levels)
        H1, W1 = pyramid1[levels].shape
        uv = cv2.resize(uv, (W1, H1))
        # uv has been upscaled, but its range also needs to be changed 
        uv = estimateHSflowlayer(pyramid1[levels], pyramid2[levels], uv, lam)
        # TODO #6: use median filter to smooth the flow result in each level in each iteration
        # 
#        uv = cv2.medianBlur(uv, (7,7))
    return uv
