import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from models.edsr import EDSR
from utils.dataset import SuperResolutionDataset
from config import Config

def main():
    # Ініціалізація конфігурації
    cfg = Config()
    
    # Підготовка даних
    dataset = SuperResolutionDataset(
        lr_dir=cfg.LR_DIR,
        hr_dir=cfg.HR_DIR,
        hr_size=(50*cfg.UPSCALE_FACTOR, 50*cfg.UPSCALE_FACTOR)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0 if cfg.DEVICE == "cpu" else 4
    )
    
    # Ініціалізація моделі
    model = EDSR(
        upscale_factor=cfg.UPSCALE_FACTOR,
        num_blocks=cfg.NUM_BLOCKS,
        channels=cfg.CHANNELS
    ).to(cfg.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.L1Loss()
    
    # Навчання
    losses = []
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(dataloader, unit="batch") as pbar:
            for lr, hr in pbar:
                lr = lr.to(cfg.DEVICE)
                hr = hr.to(cfg.DEVICE)
                
                # Forward pass
                outputs = model(lr)
                loss = criterion(outputs, hr)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # Логування
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch}/{cfg.NUM_EPOCHS}] Loss: {avg_loss:.4f}")
        
        # Збереження
        if epoch % cfg.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), f"{cfg.OUTPUT_DIR}/model_epoch_{epoch}.pth")
    
    # Фінальне збереження
    torch.save(model.state_dict(), f"{cfg.OUTPUT_DIR}/final_model.pth")
    
    # Візуалізація
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{cfg.OUTPUT_DIR}/loss_curve.png")
    plt.close()

if __name__ == "__main__":
    main()