import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import mtcnn
import cv2
import numpy as np
import tools_matrix as tools

device_id = 3
threshold = [0.4, 0.6,0.5]

if __name__=="__main__":
    image_name = "juzhao3.jpeg"
    pnet = mtcnn.Pnet()
    pnet.load_state_dict(torch.load("pnet/face_detect_1225_iter_499000_.pth"))
    pnet.eval()
    pnet = pnet.cuda(device_id)

    rnet = mtcnn.Rnet()
    rnet.load_state_dict(torch.load("rnet/face_detect_0114_iter_499000_.pth"))
    rnet.eval()
    rnet = rnet.cuda(device_id)
    print("--------------finishing loading models--------")

    #----laod images---
    image = cv2.imread(image_name) 
    original_h, original_w, ch = image.shape

    #----get scales----
    scales = tools.calculateScales(image) 
    print("scales:\t", scales)

    pnet_outs = []
    for scale in scales:
        hs = int(original_h * scale)
        ws = int(original_w * scale)
        scale_image = cv2.resize(image,(ws,hs)) / 255.0 #---resize and rescale
        scale_image = scale_image.transpose((2, 0, 1))  
        scale_image = torch.from_numpy(scale_image.copy())
        scale_image = torch.unsqueeze(scale_image, 0) #----[1, 3, W, H]
        scale_image = Variable(scale_image).float().cuda(device_id)
        conv4_1, conv4_2, _  = pnet(scale_image)  
        conv4_1 = conv4_1.cpu().data.numpy()
        conv4_2 = conv4_2.cpu().data.numpy()
        pnet_outs.append([conv4_1, conv4_2])

    image_num = len(pnet_outs)
    rectangles = []
    for i in range(image_num):
        cls_prob, roi = pnet_outs[i]
        cls_prob = cls_prob[0][0]
        print("cls_prob")
        print(cls_prob)
        roi = roi[0]
        out_w, out_h = cls_prob.shape
        out_side = max(out_w, out_h)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1/scales[i], original_w, original_h, threshold[0]) 
        rectangles.extend(rectangle)

    print(rectangles)
    rectangles = tools.NMS(rectangles, 0.7, 'iou')
    for a_rect in rectangles:
        a_rect = [int(x) for x in a_rect]
        x1, y1, x2, y2, conf = a_rect
        cv2.rectangle(image, (x1,y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("pnet.jpg", image)
