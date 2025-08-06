import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from models.edsr import EDSR
from utils.dataset import SuperResolutionDataset
from config import Config

def calculate_psnr(img1, img2):
    """Розрахунок PSNR між двома зображеннями"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    cfg = Config()
    
    # Створюємо датасет
    dataset = SuperResolutionDataset(
        lr_dir=cfg.LR_DIR,
        hr_dir=cfg.HR_DIR,
        patch_size=cfg.PATCH_SIZE,
        num_patches_per_image=cfg.NUM_PATCHES_PER_IMAGE,
        upscale_factor=cfg.UPSCALE_FACTOR
    )
    
    # Розділяємо на тренувальний та валідаційний набори
    val_size = int(len(dataset) * cfg.VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0 if cfg.DEVICE == "cpu" else 4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0 if cfg.DEVICE == "cpu" else 4
    )
    
    # Створюємо модель
    model = EDSR(
        upscale_factor=cfg.UPSCALE_FACTOR,
        num_blocks=cfg.NUM_BLOCKS,
        channels=cfg.CHANNELS
    ).to(cfg.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.L1Loss()
    
    # Логування
    train_losses = []
    val_losses = []
    val_psnrs = []
    best_val_loss = float('inf')
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print(f"Початок тренування на {len(train_dataset)} патчах")
    print(f"Валідація на {len(val_dataset)} патчах")
    print(f"UPSCALE_FACTOR: {cfg.UPSCALE_FACTOR}")
    print(f"PATCH_SIZE: {cfg.PATCH_SIZE}")
    print(f"LR розмір: {cfg.PATCH_SIZE}x{cfg.PATCH_SIZE}")
    print(f"HR розмір: {cfg.PATCH_SIZE * cfg.UPSCALE_FACTOR}x{cfg.PATCH_SIZE * cfg.UPSCALE_FACTOR}")
    
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # Тренування
        model.train()
        train_loss = 0.0
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}", unit="batch") as pbar:
            for lr, hr in pbar:
                lr = lr.to(cfg.DEVICE)
                hr = hr.to(cfg.DEVICE)
                
                outputs = model(lr)
                loss = criterion(outputs, hr)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Валідація
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for lr, hr in val_dataloader:
                lr = lr.to(cfg.DEVICE)
                hr = hr.to(cfg.DEVICE)
                
                outputs = model(lr)
                loss = criterion(outputs, hr)
                psnr = calculate_psnr(outputs, hr)
                
                val_loss += loss.item()
                val_psnr += psnr.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_psnr = val_psnr / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        
        print(f"Epoch [{epoch}/{cfg.NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {avg_val_loss:.4f} "
              f"Val PSNR: {avg_val_psnr:.2f} dB")
        
        # Збереження найкращої моделі
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{cfg.OUTPUT_DIR}/best_model.pth")
            print(f"Збережено найкращу модель (Val Loss: {best_val_loss:.4f})")
        
        # Збереження через інтервали
        if epoch % cfg.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), f"{cfg.OUTPUT_DIR}/model_epoch_{epoch}.pth")
        
        scheduler.step()
    
    # Збереження фінальної моделі
    torch.save(model.state_dict(), f"{cfg.OUTPUT_DIR}/final_model.pth")
    
    # Побудова графіків
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    
    plt.subplot(1, 3, 2)
    plt.plot(val_psnrs, label='Val PSNR')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.title("PSNR Curve")
    
    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Тренування завершено! Найкраща валідаційна втрата: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()