import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import mtcnn
import cv2
import numpy as np
import tools_matrix_torch as tools

device_id = 3
threshold = [0.4, 0.4,0.5]

if __name__=="__main__":
    image_name = "juzhao3.jpeg"
    pnet = mtcnn.Pnet()
    pnet.load_state_dict(torch.load("pnet/face_detect_190116_version3_iter_1326000_.pth"))
    pnet.eval()
    pnet = pnet.cuda(device_id)

    rnet = mtcnn.Rnet()
    rnet.load_state_dict(torch.load("rnet/face_detect_0116_iter_411000_.pth"))
    rnet.eval()
    rnet = rnet.cuda(device_id)

    onet = mtcnn.Onet()
    onet.load_state_dict(torch.load("onet/face_detect_0116_iter_751000_.pth"))
    onet.eval()
    onet = onet.cuda(device_id)
    print("--------------finishing loading models--------")

    #----laod images---
    image = cv2.imread(image_name) 
    original_h, original_w, ch = image.shape
    image_copy = image.copy()
    #----get scales----
    scales = tools.calculateScales(image) 
    print("scales")
    print(scales)

    #----pnet----
    rectangles = []
    for scale in scales:
        hs = int(original_h * scale)
        ws = int(original_w * scale)
        scale_image = cv2.resize(image,(ws,hs)) / 255.0 #---resize and rescale
        scale_image = scale_image.transpose((2, 0, 1))  
        scale_image = torch.from_numpy(scale_image.copy())
        scale_image = torch.unsqueeze(scale_image, 0) #----[1, 3, W, H]
        scale_image = Variable(scale_image).float().cuda(device_id)
        conv4_1, conv4_2, _  = pnet(scale_image)  
        cls_prob = conv4_1[0][0].cpu().data #----[1, 1, w, h] ----> [w, h]; varible to torch.tensor
        roi = conv4_2[0].cpu().data #---[1,4, w, h] -----> [4, w, h]; variable to torch.tensor
        out_w, out_h = cls_prob.size()
        out_side = max(out_w, out_h)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1.0/scale, original_w, original_h, threshold[0]) 
        rectangles.extend(rectangle)

    rectangles = tools.NMS(rectangles, 0.7, 'iou')
    """
    for a_rect in rectangles:
        a_rect = [int(x) for x in a_rect]
        x1, y1, x2, y2, conf = a_rect
        cv2.rectangle(image_copy, (x1,y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("pnet.jpg", image_copy)
    """

    #---rnet-----
    num_of_rects = len(rectangles)
    rnet_input = torch.zeros(num_of_rects, 3, 24, 24)
    for i in range(len(rectangles)):
        a_rect = rectangles[i]
        x1, y1, x2, y2 , _ = a_rect
        x1 = int(x1); y1 = int(y1)
        x2 = int(x2); y2 = int(y2)  
        crop_image = image[y1:y2, x1:x2]
        crop_image = cv2.resize(crop_image, (24, 24)) / 255.0
        crop_image = crop_image.transpose((2, 0, 1))
        crop_image = torch.from_numpy(crop_image.copy())
        rnet_input[i,:] = crop_image

    rnet_input = Variable(rnet_input).cuda(device_id)
    rnet_outputs = rnet(rnet_input)      
    fc5_1, fc5_2, fc5_3 = rnet_outputs
    fc5_1 = fc5_1.cpu().data.numpy()
    fc5_2 = fc5_2.cpu().data.numpy()
    rectangles = tools.filter_face_24net(fc5_1, fc5_2, rectangles, original_w, original_h, threshold[1])  
    """
    for a_rect in rectangles:
        a_rect = [int(x) for x in a_rect]
        x1, y1, x2, y2, conf = a_rect
        cv2.rectangle(image_copy, (x1,y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("rnet.jpg", image_copy)    
    """

    #----Onet----
    num_of_rects = len(rectangles)
    onet_input = torch.zeros(num_of_rects, 3, 48, 48)
    for i in range(len(rectangles)):
        a_rect = rectangles[i]
        x1, y1, x2, y2 , _ = a_rect
        x1 = int(x1); y1 = int(y1)
        x2 = int(x2); y2 = int(y2)
        crop_image = image[y1:y2, x1:x2]
        crop_image = cv2.resize(crop_image, (48, 48)) / 255.0
        crop_image = crop_image.transpose((2, 0, 1))
        crop_image = torch.from_numpy(crop_image.copy())
        onet_input[i,:] = crop_image

    onet_input = Variable(onet_input).cuda(device_id)
    onet_outputs = onet(onet_input)
    conv6_1, conv6_2, conv6_3 = onet_outputs
    conv6_1 = conv6_1.cpu().data.numpy()
    conv6_2 = conv6_2.cpu().data.numpy()
    conv6_3 = conv6_3.cpu().data.numpy()
    rectangles = tools.filter_face_48net(conv6_1, conv6_2, conv6_3, rectangles, original_w, original_h, threshold[2]) 
   

    for rectangle in rectangles:
        cv2.putText(image_copy, str(rectangle[4]), (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0))
        cv2.rectangle(image_copy, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255,0,0), 1)
        for i in range(5,15,2):
            cv2.circle(image_copy, (int(rectangle[i]), int(rectangle[i+1])), 2, (0, 255, 0))
    cv2.imwrite('test.jpg', image_copy)
