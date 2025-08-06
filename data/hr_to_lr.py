import cv2
import os
import numpy as np
import sys

# Додаємо кореневу папку до шляху для імпорту модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def hr_to_lr():
    cfg = Config()
    
    hr_dir = cfg.HR_DIR
    lr_dir = cfg.LR_DIR
    
    # Фіксований розмір для всіх LR зображень
    LR_SIZE = 50  # 50x50 пікселів як у попередній моделі
    
    # Створюємо папку lr якщо її немає
    os.makedirs(lr_dir, exist_ok=True)
    
    # Отримуємо список всіх файлів в hr папці
    hr_files = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Знайдено {len(hr_files)} HR зображень")
    print(f"Конвертую в LR зображення розміром {LR_SIZE}x{LR_SIZE}...")
    
    for i, img_name in enumerate(hr_files):
        # Читаємо HR зображення
        hr_path = os.path.join(hr_dir, img_name)
        hr_img = cv2.imread(hr_path)
        
        if hr_img is None:
            print(f"Помилка читання файлу: {img_name}")
            continue
        
        # Зменшуємо розмір зображення до фіксованого розміру
        lr_img = cv2.resize(hr_img, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_CUBIC)
        
        # Зберігаємо LR зображення
        # Змінюємо розширення на .png для консистентності
        base_name = os.path.splitext(img_name)[0]
        lr_filename = f"{base_name}.png"
        lr_path = os.path.join(lr_dir, lr_filename)
        
        cv2.imwrite(lr_path, lr_img)
        
        if (i + 1) % 100 == 0:
            print(f"Оброблено {i + 1}/{len(hr_files)} зображень")
    
    print(f"Конвертація завершена! Створено {len(hr_files)} LR зображень розміром {LR_SIZE}x{LR_SIZE}")
    print(f"LR зображення збережено в: {lr_dir}")

if __name__ == "__main__":
    hr_to_lr()