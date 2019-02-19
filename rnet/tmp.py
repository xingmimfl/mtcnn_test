import os
import sys
import torch
sys.path.append("..")
import mtcnn

model_path = "model_rnet_20190205/rnet_20190205_iter_1499000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "rnet_20190205_iter_1499000_.pth")
