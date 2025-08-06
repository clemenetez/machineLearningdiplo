import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from models.edsr import EDSR
from utils.dataset import SuperResolutionDataset

class Config:
    
    LR_DIR = "data/lr"
    HR_DIR = "data/hr"
    
    # Параметри для роботи з патчами
    PATCH_SIZE = 50  # Розмір LR зображень
    NUM_PATCHES_PER_IMAGE = 8
    
    # Параметри моделі EDSR (збільшення в 10 разів)
    UPSCALE_FACTOR = 10  # Збільшення в 10 разів
    NUM_BLOCKS = 16     # Більше блоків = краща якість
    CHANNELS = 64
    
    # Параметри тренування
    BATCH_SIZE = 16     # Оптимальний розмір
    NUM_EPOCHS = 200    # Більше тренування = краща якість
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Параметри збереження
    SAVE_INTERVAL = 20
    OUTPUT_DIR = "results"
    
    # Додаткові параметри
    VALIDATION_SPLIT = 0.1  # 10% для валідації