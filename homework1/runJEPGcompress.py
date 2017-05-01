import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def jpegCompress(image, quantmatrix):
    '''
        Compress(imagefile, quanmatrix simulates the lossy compression of 
        baseline JPEG, by quantizing the DCT coefficients in 8x8 blocks
    '''
    # Return compressed image in result

    H = np.size(image, 0)
    W = np.size(image, 1)

    # Number of 8x8 blocks in the height and width directions
    h8 = H / 8
    w8 = W / 8
    
    # TODO If not an integer number of blocks, pad it with zeros
    image_1 = image[:,:,0]
    image_2 = image[:,:,1]
    image_3 = image[:,:,2]
    
    add_h = (8-(H-8*h8))%8
    add_h_mat = np.zeros((add_h,W))
    temp1 = np.row_stack((image_1,add_h_mat))
    temp2 = np.row_stack((image_2,add_h_mat))
    temp3 = np.row_stack((image_3,add_h_mat))
    
    add_w = (8-(W-8*w8))%8
    height_new = np.size(temp1,0)
    add_w_mat = np.zeros((height_new,add_w))
    temp1 = np.column_stack((temp1,add_w_mat))
    temp2 = np.column_stack((temp2,add_w_mat))
    temp3 = np.column_stack((temp3,add_w_mat))
    
    # TODO Separate the image into blocks, and compress the blocks via quantization DCT coefficients
    
    H_new = np.size(temp1,0)
    W_new = np.size(temp1,1)
    h8_new = H_new/8
    w8_new = W_new/8
    for i in range(h8_new):
        for j in range(w8_new):
            block_temp1 = temp1[i*8:i*8+8,j*8:j*8+8]-128
            block_temp2 = temp2[i*8:i*8+8,j*8:j*8+8]-128
            block_temp3 = temp3[i*8:i*8+8,j*8:j*8+8]-128

            #calculat the D value
            dct_1 = cv2.dct(block_temp1)
            dct_2 = cv2.dct(block_temp2)
            dct_3 = cv2.dct(block_temp3)         

            #Quantization
            c1 = np.around((np.divide(dct_1,quantmatrix)))
            c2 = np.around((np.divide(dct_2,quantmatrix)))
            c3 = np.around((np.divide(dct_3,quantmatrix)))

   
    # TODO Convert back from DCT domain to RGB image
            R1 = np.multiply(c1,quantmatrix)
            R2 = np.multiply(c2,quantmatrix)
            R3 = np.multiply(c3,quantmatrix)
            
            decom1 = np.around(cv2.idct(R1) + 128)
            decom2 = np.around(cv2.idct(R2) + 128)
            decom3 = np.around(cv2.idct(R3) + 128)

            temp1[i*8:i*8+8,j*8:j*8+8] = decom1
            temp2[i*8:i*8+8,j*8:j*8+8] = decom2
            temp3[i*8:i*8+8,j*8:j*8+8] = decom3

    num_all = H_new*W_new
    temp1 = np.reshape(temp1, num_all, order='F')
    temp2 = np.reshape(temp2, num_all, order='F')
    temp3 = np.reshape(temp3, num_all, order='F')
    result = [temp1,temp2,temp3]
    result = np.reshape(result, num_all*3)
    result = np.reshape(result, (H_new, W_new, 3), order='F')
    
##    result = np.int32(result)
    print result.shape 
    return 255-result


if __name__ == '__main__':

    im = cv2.imread('./misc/lena_gray.bmp')
    im.astype('float')

    #check the shape of the image
    print im.shape

    
    quantmatrix = sio.loadmat('./misc/quantmatrix.mat')['quantmatrix']

    out = jpegCompress(im, quantmatrix)

    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('output',out)
    #   cv2.waitKey(0)
    plt.imshow(out)
    plt.show()

