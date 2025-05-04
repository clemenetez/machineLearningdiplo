import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from models.edsr import EDSR
from utils.dataset import SuperResolutionDataset

class Config:
    # Шляхи
    LR_DIR = "data/lr"
    HR_DIR = "data/hr"
    
    # Параметри моделі
    UPSCALE_FACTOR = 10     # 50x50 → 500x500
    NUM_BLOCKS = 8
    CHANNELS = 64
    
    # Тренування
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Збереження
    SAVE_INTERVAL = 10
    OUTPUT_DIR = "results"