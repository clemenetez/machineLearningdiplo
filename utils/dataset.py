import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, hr_size=(500, 500)):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.hr_size = hr_size
        self.image_names = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))]
        
        
        assert len(os.listdir(lr_dir)) == len(self.image_names), "LR і HR кількість не співпадає!"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
       
        lr_path = os.path.join(self.lr_dir, img_name)
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.resize(lr_img, (50, 50))  
        
        
        hr_path = os.path.join(self.hr_dir, img_name)
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.resize(hr_img, self.hr_size)  
        
        
        lr_tensor = torch.tensor(lr_img, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        hr_tensor = torch.tensor(hr_img, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        
        return lr_tensor, hr_tensor