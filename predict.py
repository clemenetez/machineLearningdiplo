import torch
import cv2
import numpy as np
from models.edsr import EDSR  # Імпортуйте вашу модель
from config import Config  # Імпортуйте конфіг

def upscale_image(model, lr_image_path, output_path, device):
    # Завантаження LR зображення
    lr_img = cv2.imread(lr_image_path)
    if lr_img is None:
        raise FileNotFoundError(f"Файл {lr_image_path} не знайдено!")

    # Підготовка тензора
    lr_tensor = torch.tensor(lr_img, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
    lr_tensor = (lr_tensor / 127.5) - 1.0  # Нормалізація [-1, 1]
    lr_tensor = lr_tensor.unsqueeze(0).to(device)  # Додаємо batch-вісь

    # Інференс
    with torch.no_grad():
        hr_tensor = model(lr_tensor)

    # Постобробка
    hr_tensor = (hr_tensor.squeeze().permute(1, 2, 0).cpu().numpy())  # CHW -> HWC
    hr_image = (hr_tensor * 127.5 + 127.5).astype(np.uint8)  # Денормалізація [0, 255]

    # Збереження
    cv2.imwrite(output_path, hr_image)
    print(f"Зображення збережено: {output_path}")

if __name__ == "__main__":
    # Налаштування
    cfg = Config()
    model_path = "results/final_model.pth"
    input_image = "0001.png"  # Ваше LR зображення
    output_image = "result_hr.jpg"

    # Завантаження моделі
    model = EDSR(
        upscale_factor=cfg.UPSCALE_FACTOR,
        num_blocks=cfg.NUM_BLOCKS,
        channels=cfg.CHANNELS
    ).to(cfg.DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()

    # Запуск апскейлу
    upscale_image(model, input_image, output_image, cfg.DEVICE)