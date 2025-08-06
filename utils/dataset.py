import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=48, hr_size=(500, 500), num_patches_per_image=4, upscale_factor=2):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.hr_size = hr_size
        self.num_patches_per_image = num_patches_per_image
        self.upscale_factor = upscale_factor
        self.image_names = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Створюємо список всіх можливих патчів
        self.patches = []
        for img_name in self.image_names:
            hr_path = os.path.join(self.hr_dir, img_name)
            hr_img = cv2.imread(hr_path)
            if hr_img is not None:
                h, w = hr_img.shape[:2]
                # Генеруємо випадкові позиції для патчів
                for _ in range(self.num_patches_per_image):
                    if h > self.patch_size * self.upscale_factor and w > self.patch_size * self.upscale_factor:
                        y = random.randint(0, h - self.patch_size * self.upscale_factor)
                        x = random.randint(0, w - self.patch_size * self.upscale_factor)
                        self.patches.append((img_name, x, y))
        
        assert len(self.patches) > 0, "Не знайдено підходящих зображень для створення патчів!"

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_name, x, y = self.patches[idx]
        
        # Завантажуємо HR зображення
        hr_path = os.path.join(self.hr_dir, img_name)
        hr_img = cv2.imread(hr_path)
        
        # Вирізаємо HR патч (більший розмір)
        hr_patch_size = self.patch_size * self.upscale_factor
        hr_patch = hr_img[y:y+hr_patch_size, x:x+hr_patch_size]
        
        # Створюємо LR патч шляхом зменшення якості HR патча
        lr_patch = cv2.resize(hr_patch, (self.patch_size, self.patch_size), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Додаємо шум для більшої реалістичності
        noise = np.random.normal(0, 5, lr_patch.shape).astype(np.uint8)
        lr_patch = np.clip(lr_patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Конвертуємо в тензори
        lr_tensor = torch.tensor(lr_patch, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        hr_tensor = torch.tensor(hr_patch, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
        
        return lr_tensor, hr_tensor