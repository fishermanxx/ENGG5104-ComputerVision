# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import pdtools
import math
from sklearn import svm
'''
This function is for extracting HOG descriptor of an image
  Input:
          1. im: A grayscale image in height x width.
          2. bins: The number of bins in histogram.
          3. cells: The number of pixels in a cell.
          4. blocks: The number of cells in a block.
  Output:
          1. HOGBlock: The HOG descriptor of the input image 
'''
def ExtractHOG(im, bins, cells, blocks):
    # Pad the im in order to make the height and width the multiplication of
    # the size of cells.
    height, width = im.shape[0], im.shape[1]

    padHeight = 0
    padWidth = 0
    if height % cells[0] != 0:
        padHeight = cells[0] - height % cells[0]

    if width % cells[1] != 0:
        padWidth = cells[1] - width % cells[1]

    im = np.pad(im, ((0, padHeight), (0, padWidth)), 'edge')
    height, width = im.shape[0], im.shape[1]

    #########################################################################
    # TODO 1: 
    #  Compute the vertical and horizontal gradients for each pixel. Put them 
    #  in gradY and gradX respectively. In addition, compute the angles (using
    #  atan2) and magnitudes by gradX and gradY, and put them in angle and 
    #  magnitude.
    ########################################################################
    hx = np.array([[-1, 0, 1]])
    hy = hx.transpose()
    gradX = np.zeros((height, width))
    gradY = np.zeros((height, width))
    angle = np.zeros((height, width))
    magnitude = np.zeros((height, width))

    ###########################  Begin TODO 1 #################################
    gradX = cv2.filter2D(im, -1, hx).astype('int32')
    gradY = cv2.filter2D(im, -1, hy).astype('int32')
    for i in range(height):
        for j in range(width):
            angle[i, j] = math.atan2(gradY[i, j], gradX[i, j])
    magnitude = np.sqrt(gradX**2 + gradY**2)

    ###########################  End TODO 1 ###################################



    #############################################################################
    # TODO 2: 
    #  Construct HOG for each cells, and put them in HOGCell. numberOfVerticalCell
    #  and numberOfHorizontalCell are the numbers of cells in vertical and 
    #  horizontal directions.
    #  You should construct the histogram according to the bins. The bins range
    #  from -pi to pi in this project, and the interval is given by
    #  (2*pi)/bins.
    ##############################################################################
    numberOfVerticalCell = height/cells[0]
    numberOfHorizontalCell = width/cells[1]
    HOGCell = np.zeros((numberOfVerticalCell, numberOfHorizontalCell, bins))

    ###########################  Begin TODO 2 #################################
    PI = math.pi
    angle_vote = (angle+PI)/(2*PI)*bins
    for i in range(numberOfVerticalCell):
        for j in range(numberOfHorizontalCell):
            start_point = np.array([i*cells[0], j*cells[1]])
            for k in range(cells[0]):
                for l in range(cells[1]):
                    bin_num = math.floor(angle_vote[start_point[0]+k, start_point[1]+l])
                    HOGCell[i, j, bin_num] += magnitude[start_point[0]+k, start_point[1]+l] 
    ###########################  End TODO 2 ###################################



    ############################################################################
    # TODO 3: 
    #  Concatenate HOGs of the cells within each blocks and normalize them. 
    #  Please remember to involve the small constant epsilon to avoid "division
    #  by zero". 
    #  The result should be stored in HOGBlock, where numberOfVerticalBlock and
    #  numberOfHorizontalBlock are the number of blocks in vertical and
    #  horizontal directions
    ###############################################################################
    numberOfVerticalBlock = numberOfVerticalCell - 1
    numberOfHorizontalBlock = numberOfHorizontalCell - 1
    HOGBlock = np.zeros((numberOfVerticalBlock, numberOfHorizontalBlock, \
                         blocks[0]*blocks[1]*bins))
    epsilon = 1e-10
    
    ###########################  Begin TODO 3 #################################
    for i in range(numberOfVerticalBlock):
        for j in range(numberOfHorizontalBlock):
            start_point = np.array([i, j])
            cell_part = []
            for k in range(blocks[0]):
                for l in range(blocks[1]):
                    cell_part.append(HOGCell[i+k, j+l])
            vec = np.hstack((x for x in cell_part))
            vec_sum = np.sqrt(np.sum(vec**2)+epsilon**2)
            vec /= vec_sum
            HOGBlock[i, j] = vec   
    ###########################  End TODO 3 ###################################
    return HOGBlock
    
'''
  This function is for training multiple components of detector
  Input:
          1. positiveDescriptor: The HOG descriptors of positive samples.
          2. negativeDescriptor: The HOG descriptors of negative samples.
  Output:
          1. detectors: Multiple components of detector
'''
def TrainMultipleComponent(positiveDescriptor, negativeDescriptor):
    ##########################################################################
    # TODO 1: 
    #  You should firstly set the number of components, e.g. 3. Then apply
    #  k-means to cluster the HOG descriptors of positive samples into 
    #  'numberOfComponent' clusters.
    ##########################################################################
    numberOfComponent = 3

    ###########################  Begin TODO 1 #################################
    rand_seed = np.random.choice(len(positiveDescriptor), numberOfComponent, replace=False)
    detector = positiveDescriptor[rand_seed, :]
    detect_class = np.zeros(len(positiveDescriptor))

    for epoch in range(20):
    	dist = []
    	for i in range(numberOfComponent):
    		temp = positiveDescriptor - detector[i]
    		dist.append(np.sum(temp**2, axis=1))

    	dist = np.stack((x for x in dist), axis=1)
    	detect_class = np.argmin(dist, axis=1)

    	old_detector = detector.copy()
    	for i in range(numberOfComponent):
    		mask = (detect_class == i)
    		temp = positiveDescriptor[mask]
    		detector[i] = np.mean(temp, axis=0)

    	diff = np.sum((old_detector - detector)**2, axis=1)
    	print diff
    	if np.max(diff) < 5e-3:
    		break

    ###########################  End TODO 1 ###################################

    
    
    ############################################################################
    # TODO 2: 
    #  After TODO 1, you have 'numberOfComponent' positive sample clusters.
    #  Then you should use train 'numberOfComponent' detectors in this TODO.
    #  For example, if 'numberOfComponent' is 3, then the:
    #    1st detector should be trained with 1st cluster of positive samples vs
    #    all negative samples;
    #    2nd detector should be trained with 2nd cluster of positive samples vs
    #    all negative samples;
    #    3rd detector should be trained with 3rd cluster of positive samples vs
    #    all negative samples;
    #    ...
    #  To train all detectors, please use SVM toolkit such as sklearn.svm.
    detectors = [None] * numberOfComponent

    ###########################  Begin TODO 2 ###########################
    detectors = []
    for i in range(numberOfComponent):
    	print '%dth detector' %i
        mask = (detect_class == i)
        temp = positiveDescriptor[mask]
        # print temp.shape
        # print negativeDescriptor.shape
        train_data = np.vstack((temp, negativeDescriptor))
        # print train_data.shape
        
        train_label = np.zeros(len(train_data))
        train_label[:len(temp)] = 1
        # print train_label.shape, sum(train_label)
    
        model = svm.SVC(kernel='linear')
        model.fit(train_data, train_label)
        param = model.coef_
        detectors.append(param.reshape((-1)))
    detectors = np.stack((x for x in detectors), axis=0)  
    ###########################  End TODO 2 ###########################
    return detectors    


def MyConv(inp, filt):
    """
    input:
    inp(H, W, C)
    filt(Hf, Wf, C)
    output:
    ans(H + 1 - Hf, W + 1 - Wf)
    """
    H, W, C = inp.shape
    Hf, Wf, _ = filt.shape
#    print 'H:', H, '| W:', W, '| C:', C, '| Hf:', Hf, '| Wf:', Wf
    
    if H < Hf:
        padH = np.zeros((Hf - H, W, C))
        inp = np.vstack((inp, padH))
        H = Hf
    
    if W < Wf:
        padW = np.zeros((H, Wf - W, C))
        inp = np.hstack((inp, padW))
        W = Wf
        
    
    ans = np.zeros((H + 1 - Hf, W + 1 - Wf))
    for i in range(H + 1 - Hf):
        for j in range(W + 1 - Wf):
            temp = inp[i:i+Hf, j:j+Wf, :]
            ans[i, j] = np.sum(temp*filt)
            
    return ans




'''
  This function is for multiscale detection
  Input:
          1. im: A grayscale image in height x width.
          2. detectors: The trained linear detectors. Each one is in 
                        row x col x dim. The third dimension 'dim' should 
                        be the same with the one of the HOG descriptor 
                        extracted from input image im.
          3. threshold: The constant threshold to control the prediction.
          4. bins: The number of bins in histogram.
          5. cells: The number of pixels in a cell.
          6. blocks: The number of cells in a block.
  Output:
          1. bbox: The predicted bounding boxes in this format (n x 5 matrix):
                                   x11 y11 x12 y12 s1
                                   ... ... ... ... ...
                                   xi1 yi1 xi2 yi2 si
                                   ... ... ... ... ...
                                   xn1 yn1 xn2 yn2 sn
                   where n is the number of bounding boxes. For the ith 
                   bounding box, (xi1,yi1) and (xi2, yi2) correspond to its
                   top-left and bottom-right coordinates, and si is the score
                   of convolution. Please note that these coordinates are
                   in the input image im.
'''



def MultiscaleDetection(im, detectors, threshold, bins, cells, blocks):
    #############################################################################
    # TODO 1: 
    #  You should firstly generate a series of scales, e.g. 0.5, 1, 2. And then
    #  resize the input image by scales and store them in the structure pyra.
    ############################################################################
    pyra = []
#    scales = [0.3, 0.5, 0.8, 1, 1.5, 2]
    scales = [0.5, 1, 2]

    ###########################  Begin TODO 1 ###########################

    for i in range(len(scales)):
    	temp = im
    	H_new = int(temp.shape[0]*scales[i])
    	W_new = int(temp.shape[1]*scales[i])
    	temp2 = cv2.resize(temp, (W_new, H_new))
    	pyra.append(temp2)
    ###########################  End TODO 1 ###########################


    #############################################################################
    #  TODO 2:
    #  Perform detection on multiscale. Please remember to transfer the
    #  coordinates of bounding box according to their scales
    #############################################################################
    bbox = []
    numberOfScale = len(pyra)

    ###########################  Begin TODO 2 ###########################
    for i in range(numberOfScale):
    	im_feature = ExtractHOG(pyra[i], bins, cells, blocks)
    	for j in range(len(detectors)):

    		result = MyConv(im_feature, detectors[j])
    		filt_h, filt_w = detectors[j].shape[0], detectors[j].shape[1]
    		mask = (result > threshold)
    		index = np.where(mask)
    		block_x, block_y = index[0], index[1]
    		x_1, y_1 = block_x*cells[0]/scales[i] , block_y*cells[1]/scales[i]
    		x_2, y_2 = (block_x + filt_h)*cells[0]/scales[i], (block_y+filt_w)*cells[1]/scales[i]
    		s = result[mask].reshape(-1)
    		for k in range(len(x_1)):
#    			temp = np.array([int(x_1[k]), int(y_1[k]), int(x_2[k]), int(y_2[k]), s[k]])
    			temp = np.array([int(y_1[k]), int(x_1[k]), int(y_2[k]), int(x_2[k]), s[k]])
    			bbox.append(temp)


    bbox = np.stack((i for i in bbox), axis=0)
    ###########################  End TODO 2 ###########################
    return bbox    
    
    
# Set the number of bin to 9
bins = 9

# Set the size of cell to cover 8 x 8 pixels
cells = [8, 8]

# Set the size of block to contain 2 x 2  cells
blocks = [2, 2]    
    
###################################################################
# Step 1: Extract HOG descriptors of postive and negative samples
###################################################################
pos = np.load('./Dataset/Train/pos.npy')
neg = np.load('./Dataset/Train/neg.npy')
numberOfPositive = pos.shape[3]
numberOfNegative = neg.shape[3]
height, width = pos.shape[0], pos.shape[1]

# Delete the descriptor files if you want to extract new ones
if os.path.exists('./Descriptor/positiveDescriptor.npy') and \
    os.path.exists('./Descriptor/negativeDescriptor.npy'):
        
    positiveDescriptor = np.load('./Descriptor/positiveDescriptor.npy')
    negativeDescriptor = np.load('./Descriptor/negativeDescriptor.npy')
    print 'Load Descriptor successfully!'
else:
    positiveDescriptor = [None] * numberOfPositive
    for ii in range(0, numberOfPositive):
        print('Positive HOG descriptor: ' + str(ii + 1) + '\\' + str(numberOfPositive))
        temp = ExtractHOG(cv2.cvtColor(pos[:,:,:,ii], cv2.COLOR_BGR2GRAY), bins, cells, blocks)
        positiveDescriptor[ii] = temp.ravel().transpose()
    positiveDescriptor = np.array(positiveDescriptor)
    
    negativeDescriptor = [None] * numberOfNegative
    for ii in range(0, numberOfNegative):
        print('Negative HOG descriptor: ' + str(ii + 1) + '\\' + str(numberOfNegative))
        temp = ExtractHOG(cv2.cvtColor(neg[:,:,:,ii], cv2.COLOR_BGR2GRAY), bins, cells, blocks)
        negativeDescriptor[ii] = temp.ravel().transpose()   
    negativeDescriptor = np.array(negativeDescriptor)
    
    np.save('./Descriptor/positiveDescriptor.npy', positiveDescriptor)
    np.save('./Descriptor/negativeDescriptor.npy', negativeDescriptor)
    print 'save successfully!'


########################################################################  
# Step 2: Train Linear Detector
########################################################################
# Delete the detector file if you want to train a new one
if os.path.exists('./Detector/detectors.npy'):
    detectors = np.load('./Detector/detectors.npy')
    print 'Load detector successfully!'
else:
    print('Training linear detector');
    detectors = TrainMultipleComponent(positiveDescriptor, negativeDescriptor);
    # for ii in range(0, len(detectors)):
    #     detectors[ii] = np.reshape(detectors[ii], (height/cells[0] - 1, \
    #                             width/cells[1] - 1, blocks[0] * blocks[1] * bins))
    detect_record = detectors.reshape((detectors.shape[0], height/cells[0] - 1, width/cells[1] - 1, blocks[0] * blocks[1] * bins))

    np.save('.\Detector\detectors.npy', detect_record)
    print 'save detector successfully!'  


#########################################################################    
# Step 3: Detection
#########################################################################
validation = np.load('./Dataset/Validation/validation.npy')
groundTruth = np.load('./Dataset/Validation/groundTruth.npy')

numPositives = 0;
for ii in range(0, len(groundTruth)):
    numPositives = numPositives + groundTruth[ii].shape[0];

Label = [None] * len(validation)
Score = [None] * len(validation)

for ii in range (0, len(validation)):
# for ii in range (0, 5):
    print('Detect ' + str(ii) + '...')
    bbox = MultiscaleDetection(cv2.cvtColor(validation[ii], cv2.COLOR_BGR2GRAY), \
                               detectors, 6, bins, cells, blocks)  
    top = np.arange(bbox.shape[0])
    # Non-maximum suppression. Uncomment this line if you want this
    # process.
    # top = pdtools.NonMaxSup(bbox, 0.5)

    # Measure the performance
    labels, scores = pdtools.MeasureDetection(bbox[top, :], groundTruth[ii], 0.5)
    Label[ii] = labels;
    Score[ii] = scores;
        
    # Show the bounding boxes. Uncomment the following two line if you 
    # want to show the bounding boxes.
    pdtools.ShowBoxes(validation[ii], bbox[:,:])

    raw_input("Press Enter to continue...")

Label = np.array(Label)
Score = np.array(Score)
ap = pdtools.DrawPRC(Label, Score, numPositives)
