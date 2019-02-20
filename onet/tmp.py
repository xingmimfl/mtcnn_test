import os
import sys
import torch
sys.path.append("..")
import mtcnn

model_path = "model_face_detect_0219/face_detect_0219_iter_1462000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "face_detect_0219_iter_1462000_.pth")
