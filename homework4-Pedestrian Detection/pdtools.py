# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Malisiewicz et al.
def NonMaxSup(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes	
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

    
def OverlapRate(testBox, targetBox):
    rate = -np.inf
    bi = [np.maximum(testBox[0], targetBox[0]), np.maximum(testBox[1], targetBox[1]), \
          np.minimum(testBox[2], targetBox[2]), np.minimum(testBox[3], targetBox[3])]

    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    
    if iw > 0 and ih > 0:                
        # compute overlap as area of intersection / area of union
        ua = (testBox[2] - testBox[0] + 1 ) * (testBox[3] - testBox[1] + 1) + \
             (targetBox[2] - targetBox[0] + 1) * (targetBox[3] - targetBox[1] + 1) - \
             iw * ih
    
        ov = iw * ih / ua
        if ov > rate:
            rate = ov
    return rate
    
'''    
This function is for measuring the performance.
  Input:
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
          2. groundTruth: The provided ground truth bounding boxes. It is                   
          in this format:
                                   x11 y11 x12 y12
                                   ... ... ... ...
                                   xi1 yi1 xi2 yi2
                                   ... ... ... ...
                                   xm1 ym1 xm2 ym2
                   where m is the number of groundtruth bounding boxes.
          3. overlapThreshold: The overlapping rate which decides the true
                   detection (default 0.5).
  Output:
          1. labels: The labels of the bounding box (1 for true hypothesis 
                   and -1 for false)
          2. scores: The scores of the bounding box
'''
def MeasureDetection(bbox, groundTruth, overlapThreshold):
    numberOfBoundingBox = bbox.shape[0]
    numberOfGroundTruth = groundTruth.shape[0]
    overRate = np.zeros((numberOfGroundTruth, numberOfBoundingBox))

    for ii in range(0, numberOfGroundTruth):
        for jj in range(0, numberOfBoundingBox):
            overRate[ii, jj] = OverlapRate(bbox[jj, 0:4], groundTruth[ii, :])

    labels = -np.ones((numberOfBoundingBox, 1))
    scores = bbox[:, 4]
    for ii in range(0, numberOfGroundTruth):
        idx = np.where(overRate[ii, :] >= overlapThreshold)
        idx = idx[0]
        ovmax = -np.inf
        jmax = 0
        for jj in range(0, len(idx)):
            if overRate[ii, idx[jj]] > ovmax:
                ovmax = overRate[ii, idx[jj]]
                jmax=idx[jj];
        
    if jmax >= 0:
        labels[jmax] = 1;

    return labels, scores

def ShowBoxes(im, boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    for i in range(0, boxes.shape[0]):
        # Create a Rectangle patch
        rect = patches.Rectangle((boxes[i, 0], boxes[i, 1]), boxes[i, 2] - boxes[i, 0], \
                                 boxes[i, 3] - boxes[i, 1] ,linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
    
def DrawPRC(Label, Score, numPositives):
    tp = np.zeros((1, len(Label)))
    fp = np.zeros((1, len(Label)))
    tp[Label == 1] = 1
    fp[Label == -1] = 1

    idx = np.argsort(Score)[::-1]
    fp = fp[idx]
    tp = tp[idx]
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / numPositives
    prec = tp / (fp + tp)
    
    mrec = np.r_[0, rec, 1]
    mpre = np.r_[0, prec, 0]

    
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[0:-1]) 
    i = i[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    fig = plt.figure()
    fig.suptitle('AP = ' + str(ap))
    ax = fig.add_subplot(111)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    plt.plot(rec, prec)
	
    return ap