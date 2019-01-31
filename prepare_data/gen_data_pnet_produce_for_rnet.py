# -*- coding: utf-8 -*-  
#----用Pnet的模型对wider face图片进行预测，把其中产生的样本当做hard example再次进行训练
#----尤其是本来是正样本预测成负样本的
#----阈值初步选定为0.5
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
from utils import ensure_directory_exists
sys.path.insert(0, "..")
import tools_matrix_torch as tools
import mtcnn
import torch
from torch.autograd import Variable

device_id = 0
threshold = [0.2, 0.6, 0.6]
pnet = mtcnn.Pnet()
pnet.load_state_dict(torch.load("../pnet/face_detect_power_bbox2_retrain_iter_999000_.pth"))
pnet.eval()
pnet = pnet.cuda(device_id)

anno_file = "wider_face_train.txt"
im_dir = "/media/disk1/mengfanli/new-caffe-workplace/MTCNN_workplace/mtcnn-caffe_without_landmarks/prepare_data/WIDER_train/images"
pos_save_dir = "../rnet/24/positive_hardmining"
part_save_dir = "../rnet/24/part_hardmining"
neg_save_dir = '../rnet/24/negative_hardmining'
save_dir = "../rnet/24"

ensure_directory_exists(save_dir)
ensure_directory_exists(pos_save_dir)
ensure_directory_exists(neg_save_dir)
ensure_directory_exists(part_save_dir)

f1 = open(os.path.join(save_dir, 'pos_24_hardmining.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_24_hardmining.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_24_hardmining.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = [float(x) for x in annotation[1:]]
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    image = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    print(im_path + '.jpg')
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")

    height, width, channel = image.shape

    #----pnet predict process----
    scales = tools.calculateScales(image)
    rectangles = []
    for scale in scales:
        hs = int(height * scale)
        ws = int(width * scale)
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
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1.0/scale, width, height, threshold[0])
        rectangles.extend(rectangle)
        
    rectangles = np.asarray(rectangles)

    #print(im_path)
    #a_image_name = im_path.split("/")[-1] + ".jpg"
    #a_image_path = os.path.join("plot_images", a_image_name)
    #for a_rect in rectangles:
    #    a_rect_x1, a_rect_y1, a_rect_x2, a_rect_y2 , _ =  a_rect
    #    cv2.rectangle(image, (int(a_rect_x1), int(a_rect_y1)), (int(a_rect_x2), int(a_rect_y2)), (0, 255, 0), 1)
        
    #cv2.imwrite(a_image_path, image)
    for a_rect in rectangles:
        a_rect_x1, a_rect_y1, a_rect_x2, a_rect_y2 , confidence =  a_rect
        if max((a_rect_x2 - a_rect_x1), (a_rect_y2 - a_rect_y1)) < 40:
            continue
        a_rect_width = a_rect_x2 - a_rect_x1 + 1
        a_rect_height = a_rect_y2 - a_rect_y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(a_rect_width, a_rect_height) < 40 or a_rect_x1 < 0 or a_rect_y1 < 0:
            continue


        iou = IoU(a_rect, boxes)
        max_id = np.argmax(iou)
        x1, y1, x2, y2 = boxes[max_id]
        offset_x1 = (x1 - a_rect_x1) / float(a_rect_width)
        offset_y1 = (y1 - a_rect_y1) / float(a_rect_height)
        offset_x2 = (x2 - a_rect_x2) / float(a_rect_width)
        offset_y2 = (y2 - a_rect_y2) / float(a_rect_height)

        cropped_im = image[int(a_rect_y1) : int(a_rect_y2), int(a_rect_x1) : int(a_rect_x2), :]
        resized_im = cv2.resize(cropped_im, (24, 24), interpolation=cv2.INTER_LINEAR)

        if np.max(iou) >= 0.65:
            save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
            f1.write("24/positive_hardmining/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
            cv2.imwrite(save_file, resized_im)
            p_idx += 1                
        elif np.max(iou) >=0.4:
            if confidence >= 0.5:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("24/part_hardmining/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        elif np.max(iou) < 0.3: #--negative samples
            if confidence >= 0.6:
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write("24/negative_hardmining/%s.jpg"%n_idx + ' 0 -1 -1 -1 -1\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1                    

f1.close()
f2.close()
f3.close()
