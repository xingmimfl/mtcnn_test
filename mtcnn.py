import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_LANDMARK = True

class Pnet(nn.Module):
    def __init__(self, training=False):
        super(Pnet, self).__init__()
        
        self.training = training 
        self.basenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0, bias=True),
            nn.PReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True),
            nn.RReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0,  bias=True),
            nn.PReLU(),
        )
        
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1, padding=0, bias=True)
        if USE_LANDMARK:
            self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0, bias=True)

        self.loss_func = Lossfunc()

    def forward(self, x):
        """
        conv4_1: [batch_size, 1, 1, 1]
        conv4_2: [batch_size, 4, 1, 1]
        conv4_3: [batch_size, 10, 1, 1]
        """
        x = self.basenet(x)
        conv4_1 = self.conv4_1(x)
        conv4_1 = F.sigmoid(conv4_1) #---in pytorch, we have to add sigmoid function,  for nn.BCELoss()
        conv4_2 = self.conv4_2(x)
        if USE_LANDMARK:
            conv4_3 = self.conv4_3(x)
            return [conv4_1, conv4_2, conv4_3]
        return [conv4_1, conv4_2]


    def get_loss(self, x, bbox, cls_labels, flag):
        cls_loss, bbox_loss, landmark_loss = self.loss_func(x, bbox, cls_labels, flag)
        return cls_loss, bbox_loss, landmark_loss

class Rnet(nn.Module):
    def __init__(self):
        super(Rnet, self).__init__()
        self.basenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=0, bias=True),
            nn.PReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1, padding=0, bias=True), 
            nn.PReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1, padding=0, bias=True),
            nn.PReLU(), 
        )
        self.fc4 = nn.Linear(in_features=64*3*3, out_features=128)
        self.fc5_1 = nn.Linear(in_features=128, out_features=1)
        self.fc5_2 = nn.Linear(in_features=128, out_features=4)
        self.fc5_3 = nn.Linear(in_features=128, out_features=10)


        self.loss_func = Lossfunc()

    def forward(self, x):
        x = self.basenet(x)
        #print("x.size():\t", x.size())
        x = x.view(x.size(0), -1)
        x = self.fc4(x)
        fc5_1 = self.fc5_1(x)
        fc5_1 = F.sigmoid(fc5_1)
        fc5_2 = self.fc5_2(x)
        fc5_3 = self.fc5_3(x)  
        #print("fc5_1.size:\t", fc5_1.size())
        #print("fc5_2.size:\t", fc5_2.size())
        #print("fc5_3.size:\t", fc5_3.size())
        return fc5_1, fc5_2, fc5_3


    def get_loss(self, x, bbox, cls_labels, flag):
        cls_loss, bbox_loss, landmark_loss = self.loss_func(x, bbox, cls_labels, flag)
        return cls_loss, bbox_loss, landmark_loss


class Onet(nn.Module):
    def __init__(self):
        super(Onet, self).__init__()
        self.basenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.PReLU(),             
            
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.PReLU(),                         

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.PReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0, bias=True),
            nn.PReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3, out_features=256),
            nn.Dropout(0.25),
            nn.PReLU(),            
        ) 

        self.conv6_1 = nn.Linear(in_features=256, out_features=1)
        self.conv6_2 = nn.Linear(in_features=256, out_features=4)
        self.conv6_3 = nn.Linear(in_features=256, out_features=10)

        self.loss_func = Lossfunc()

    def forward(self, x):
        x = self.basenet(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        conv6_1 = self.conv6_1(x)
        conv6_1 = F.sigmoid(conv6_1)
        conv6_2 = self.conv6_2(x)
        conv6_3 = self.conv6_3(x)
        return conv6_1, conv6_2, conv6_3

    def get_loss(self, x, bbox, cls_labels, flag):
        cls_loss, bbox_loss, landmark_loss = self.loss_func(x, bbox, cls_labels, flag)
        return cls_loss, bbox_loss, landmark_loss


class Lossfunc(nn.Module):
    def __init__(self, cls_factor=1, bbox_factor=0.5, landmark_factor=0.5):
        super(Lossfunc, self).__init__()
        self.cls_factor = cls_factor
        self.bbox_factor = bbox_factor
        self.landmark_factor = landmark_factor
        self.cls_loss_func = nn.BCELoss()
        self.bbox_loss_func = nn.MSELoss()
        self.landmark_loss_func = nn.MSELoss()

    def forward(self, x, bbox, cls_labels, flag):
        """
        x: [conv4_1, conv4_2, conv4_3]
            conv4_1: [batch_size, 1, 1, 1]
            conv4_2: [batch_size, 4, 1, 1]
            conv4_3: [batch_size, 10, 1,1] 
        bbox: [batch_size, 4 or 10]
        cls_labels: [batch_size, 1]
        flag: decide which loss. although we can use bbox and cls_labels decide which loss to use
            0: classification loss
            1: bbox loss
            2: landmark loss
         """
        conv4_1, conv4_2 = x[:2]
        if len(conv4_1.size())==4:
            conv4_1 = conv4_1[:, :, 0, 0] #---size: [batch_size, 1]
        if len(conv4_2.size())==4:
            conv4_2 = conv4_2[:, :, 0, 0] #---size: [batch_size, 4]

        if USE_LANDMARK:
            conv4_3 = x[2]
            if len(conv4_3.size())==4:
                conv4_3 = conv4_3[:, :, 0, 0]
        
        cls_loss=None; bbox_loss=None; landmark_loss=None 
        if flag==0:
            cls_loss = self.cls_loss(conv4_1, cls_labels)
        elif flag==1:
            bbox_loss = self.bbox_loss(conv4_2, bbox)
        elif flag==2:
            landmark_loss = self.landmark_loss(conv4_3, bbox)

        return cls_loss, bbox_loss, landmark_loss

    def cls_loss(self, x, cls_labels):
        """
        x: [batch_size, 2]
        cls_labels: [batch_size,1]
        """ 
        loss = self.cls_loss_func(x, cls_labels)     
        return loss

    def bbox_loss(self, x, bbox):
        """
        x: [batch_size, 4]
        bbox: [batch_size, 4]
        """
        loss = self.bbox_loss_func(x, bbox) / 2.0
        return loss


    def landmark_loss(self, x, bbox):
        """
        x: [batch_size, 4]
        bbox: [batch_size, 10]
        """
        loss = self.landmark_loss_func(x, bbox) / 2.0
        return loss
