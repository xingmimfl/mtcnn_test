import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
sys.path.append("..")
from config import *
import dataset
import mtcnn

def main():
    with torch.cuda.device(DEVICE_IDS[0]):
        o_model = _model_init()
        #----data loader----
        cls_train_loader = get_dataset(files_vec=['pos_48.txt', 'neg_48.txt'])
        bbox_train_loader = get_dataset(files_vec=['pos_48.txt', 'part_48.txt'])
        landmark_train_loader = get_dataset(files_vec=['landmark_48_aug.txt'])

        #----data iter-----
        cls_train_iter = iter(cls_train_loader)
        bbox_train_iter = iter(bbox_train_loader)
        landmark_train_iter = iter(landmark_train_loader)

        check_dir(SNAPSHOT_PATH)

        #----get parameters that need back-propagation
        params = []
        for p in list(o_model.parameters()):
            if p.requires_grad == False: continue
            params.append(p)

        #----optimizer
        optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        #----training
        loss_avg = AverageMeter()
        cls_loss_avg = AverageMeter()
        bbox_loss_avg = AverageMeter()
        landmark_loss_avg = AverageMeter()
        for i in range(MAX_ITERS):
            scheduler.step() 

            #---generate flag to decide which loss
            flag = random.randint(0, 2)

            if flag==0:
                try:
                    _images, _bbox, _labels, _image_paths = cls_train_iter.next()
                except StopIteration:
                    cls_train_iter = iter(cls_train_loader)
                    _images, _bbox, _labels, _image_paths = cls_train_iter.next() 
            elif flag== 1:
                try:
                    _images, _bbox, _labels, _image_paths = bbox_train_iter.next()
                except StopIteration:
                    bbox_train_iter = iter(bbox_train_loader) 
                    _images, _bbox, _labels, _image_paths = bbox_train_iter.next() 
            else:
                try:
                    _images, _bbox, _labels, _image_paths = landmark_train_iter.next()
                except StopIteration:
                    landmark_train_iter = iter(landmark_train_loader) 
                    _images, _bbox, _labels, _image_paths = landmark_train_iter.next()
                
 
            #-----training, get loss, and back-propagation
            _images = Variable(_images.cuda(DEVICE_IDS[0]))
            _bbox = Variable(_bbox.cuda(DEVICE_IDS[0]))
            _labels = Variable(_labels.cuda(DEVICE_IDS[0]))
            outputs = o_model(_images) #----model forward 
            cls_loss, bbox_loss, landmark_loss = o_model.get_loss(outputs, _bbox, _labels, flag)

            loss = 0
            #if cls_loss is not None: 
            if flag==0:
                loss += cls_loss
                cls_loss_avg.update(cls_loss.data[0], BATCH_SIZE)

            #if bbox_loss is not None:
            if flag==1:
                loss += bbox_loss
                bbox_loss_avg.update(bbox_loss.data[0], BATCH_SIZE)

            if flag==2:
                loss += landmark_loss
                landmark_loss_avg.update(landmark_loss.data[0], BATCH_SIZE)

            loss_avg.update(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i >=TRAIN_OUT_ITER)  and (i % TRAIN_OUT_ITER == 0):  # print every 2000 mini-batches
                print("iter:%5d " % i, " loss:%.4e" % loss_avg.avg, " cls_loss:%.4e" % cls_loss_avg.avg, " bbox_loss:%.4e" % bbox_loss_avg.avg, " landmark_loss:%.4e" % landmark_loss_avg.avg)
    
                save_name = '_'.join([SUFFIX, "iter", str(i), '.model'])                
                torch.save(o_model, os.path.join(SNAPSHOT_PATH, save_name))

                loss_avg = AverageMeter()
                cls_loss_avg = AverageMeter()
                bbox_loss_avg = AverageMeter()
                landmark_loss_avg = AverageMeter()
                 
def get_dataset(files_vec=None, images_vec=None):
    trainset = dataset.ImageSets(isTrain=True, imageSize=24, files_vec=files_vec, images_vec=images_vec)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=dataset.detection_collate, 
                                                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    print("length of trainset:\t", len(trainset))
    return train_loader

     
def _model_init():
    o_model = mtcnn.Onet()
    #o_model.apply(weight_init)
    o_model.cuda(DEVICE_IDS[0])
    o_model.train()
    return o_model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=="__main__":
    main()