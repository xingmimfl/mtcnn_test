import os
import sys
import torch
sys.path.append("..")
import mtcnn

model_path = "model_combining_dataloader_190204_retrain_retrain/combining_dataloader_190204_retrain_retrain_iter_1495000_.model"
model = torch.load(model_path)
torch.save(model.state_dict(), "combining_dataloader_190204_retrain_retrain_iter_1495000_.pth")
