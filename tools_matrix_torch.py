import sys
from operator import itemgetter
import numpy as np
import cv2
import torch

'''
Function:
	change rectangles into squares (matrix version)
Input:
	rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
	squares: same as input
'''
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles

'''
Function:
    change rectangles into squares (matrix version)
Input:
    rectangles:  torch.Tensor
        rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
    squares: same as input
'''
def rect2square_torch(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l, max_index = torch.max(torch.stack([w, h], dim=1), dim=1)
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
    rectangles[:,2:4] = rectangles[:,0:2] + l.view(-1, 1).repeat(1, 2)
    return rectangles
'''
Function:
	apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
Input:
	rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
	rectangles: same as input
'''
def NMS(rectangles,threshold,type):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'iom':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

def NMS_torch(rectangles, threshold, type):
    """
    Function:
        apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
    Input:
        rectangles: torch.floattensor. rectangles[i][0:3] is the position, rectangles[i][4] is score
    Output:
        rectangles: same as input
    """
    if rectangles.size(0)==0:
        return rectangles

    x1 = rectangles[:, 0]
    y1 = rectangles[:, 1]
    x2 = rectangles[:, 2]
    y2 = rectangles[:, 3]
    score = rectangles[:, 4]
    area = torch.mul(x2 - x1 + 1.0, y2 - y1 + 1)  
    _, order = torch.sort(score, descending=True)
    keep = []
    while order.size(0) > 0:
        i = order[0] 
        keep.append(i)
         
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    """
    Function:
        Detect face position and calibrate bounding box on 12net feature map(matrix version)
    Input:
        cls_prob : softmax feature map for face classify
        roi      : feature map for regression
        out_side : feature map's largest size
        scale    : current input image scale in multi-scales
        width    : image's origin width
        height   : image's origin height
        threshold: 0.6 can have 99% recall rate
    """
    stride = 2

    binary_tensor = (cls_prob >= threshold)
    indexes = binary_tensor.nonzero()
    if indexes.sum() <= 0: return []
    indexes = indexes.float() #---torch.LongTensor to torch.floatTensor, torch.round not support longTensor
    bb1 = torch.round((stride * indexes + 0) * scale)
    bb2 = torch.round((stride * indexes + 12) * scale) 
    boundingbox = torch.cat([bb1, bb2], dim=1)
    
    #dx1 = roi[0][binary_tensor].unsqueeze(1)
    #dx2 = roi[1][binary_tensor].unsqueeze(1)
    #dx3 = roi[2][binary_tensor].unsqueeze(1)
    #dx4 = roi[3][binary_tensor].unsqueeze(1)
    #offset = torch.cat([dx1, dx2, dx3, dx4], dim=1)

    dx1 = roi[0][binary_tensor]
    dx2 = roi[1][binary_tensor]
    dx3 = roi[2][binary_tensor]
    dx4 = roi[3][binary_tensor]
    offset = torch.stack([dx1, dx2, dx3, dx4], dim=1)

    score = cls_prob[binary_tensor].unsqueeze(1)
    boundingbox = boundingbox + offset * 12.0 * scale
    rectangles = torch.cat([boundingbox, score], dim=1) 
    
    rectangles = rect2square_torch(rectangles)

    zeros_tensor = torch.zeros([1])
    width_tensor = torch.Tensor([width]).float()
    height_tensor = torch.Tensor([height]).float()

    rectangles[:, :2] = torch.max(zeros_tensor, rectangles[:, :2]).int() 
    rectangles[:, 2] = torch.min(width_tensor, rectangles[:, 2]).int()
    rectangles[:, 3] = torch.min(height_tensor, rectangles[:, 3]).int() 

    index1 = (rectangles[:, 2] >= rectangles[:, 0]) #---x2 > x1
    index2 = (rectangles[:, 3] >= rectangles[:, 1]) #---y2 > y1
    index = (index1 & index2).nonzero().squeeze()
    rectangles = rectangles.index_select(0, index)
    rectangles = rectangles.numpy().tolist()
    return NMS(rectangles,0.5,'iou')


def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    """
    Function:
        Filter face position and calibrate bounding box on 12net's output
    Input:
        cls_prob  : softmax feature map for face classify. size: [N, 1]
        roi_prob  : feature map for regressiona, size:[N, 4]
        rectangles: 12net's predict
        width     : image's origin width
        height    : image's origin height
        threshold : 0.6 can have 97% recall rate
    Output:
        rectangles: possible face positions
    """
    prob = cls_prob
    binary_tensor = (prob>=threshold)
    indexes = binary_tensor.nonzero()[0] #---rows
    if indexes.sum() <= 0: return []
    rectangles = rectangles.index_select(0, indexes)
    roi = roi.index_select(0, indexes)

    w = (rectangles[:, 2] - rectangles[:, 0]).view(-1, 1) #---[N]--->[N, 1]
    h = (rectangles[:, 3] - rectangles[:, 1]).view(-1, 1)

    roi[:, 0::2] = torch.mul(roi[:, 0::2], w)
    roi[:, 1::2] = torch.mul(roi[:, 1::2], h)
    rectangles[:, :4] = rectangles[:, :4] + roi
    rectangles = rect2square_torch(rectangles)
    
    zeros_tensor = torch.zeros([1])
    width_tensor = torch.Tensor([width]).float()
    height_tensor = torch.Tensor([height]).float()

    rectangles[:, :2] = torch.max(zeros_tensor, rectangles[:, :2]).int()
    rectangles[:, 2] = torch.min(width_tensor, rectangles[:, 2]).int()
    rectangles[:, 3] = torch.min(height_tensor, rectangles[:, 3]).int()

    index1 = (rectangles[:, 2] >= rectangles[:, 0]) #---x2 > x1
    index2 = (rectangles[:, 3] >= rectangles[:, 1]) #---y2 > y1
    index = (index1 & index2).nonzero().squeeze()
    rectangles = rectangles.index_select(0, index)
    rectangles = rectangles.numpy().tolist()
    return NMS(rectangles,0.7,'iou')

def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    """
    Function:
        Filter face position and calibrate bounding box on 12net's output
    Input:
        cls_prob  : cls_prob[1] is face possibility
        roi       : roi offset
        pts       : 5 landmark
        rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
        width     : image's origin width
        height    : image's origin height
        threshold : 0.7 can have 94% recall rate on CelebA-database
    Output:
        rectangles: face positions and landmarks
    """
    prob = cls_prob
    binary_tensor = (prob>=threshold)
    indexes = binary_tensor.nonzero()[0]

    rectangles = rectangles.index_select(0, indexes)
    roi = roi.index_select(0, indexes)
    w = (rectangles[:, 2] - rectangles[:, 0]).view(-1, 1)
    h = (rectangles[:, 3] - rectangles[:, 1]).view(-1, 1)

    roi[:, 0::2] = torch.mul(roi[:, 0::2], w)
    roi[:, 1::2] = torch.mul(roi[:, 1::2], h)
    rectangles[:, :4] = rectangles[:, :4] + roi

    pts[:, 0::2] = torch.mul(pts[:, 0::2], w)
    pts[:, 0::2] = torch.add(pts[:, 0::2], rectangles[:, 0].view(-1,1)) 
    pts[:, 1::2] = torch.mul(pts[:, 1::2], h)
    pts[:, 1::2] = torch.add(pts[:, 1::2], rectangles[:, 1].view(-1,1))

    #-----
    zeros_tensor = torch.zeros([1])
    width_tensor = torch.Tensor([width]).float()
    height_tensor = torch.Tensor([height]).float()

    rectangles[:, :2] = torch.max(zeros_tensor, rectangles[:, :2]).int()
    rectangles[:, 2] = torch.min(width_tensor, rectangles[:, 2]).int()
    rectangles[:, 3] = torch.min(height_tensor, rectangles[:, 3]).int()

    index1 = (rectangles[:, 2] >= rectangles[:, 0]) #---x2 > x1
    index2 = (rectangles[:, 3] >= rectangles[:, 1]) #---y2 > y1
    
    #---concat rectangles and pts
    rectangles = torch.cat([rectangles, pts], dim=1) 
    rectangles = rectangles.numpy()
    return NMS(rectangles,0.7,'iom')

def calculateScales(img):
    """
    Function:
        calculate multi-scale and limit the maxinum side to 1000
    Input:
        img: original image
    Output:
        pr_scale: limit the maxinum side to 1000, < 1.0
        scales  : Multi-scale
    """
    caffe_img = img.copy()
    h, w, c = caffe_img.shape
    #multi-scale
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales
