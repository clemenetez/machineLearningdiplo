import cv2
import os
import numpy as np
import random
from tqdm import tqdm

def create_test_patches(input_dir, output_dir, patch_size=48, num_patches_per_image=10):
    """
    Створює тестові патчі з високої якості зображень для демонстрації
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Знаходимо всі зображення
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    print(f"Знайдено {len(image_files)} зображень")
    
    patch_count = 0
    
    for img_file in tqdm(image_files, desc="Створення патчів"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Перевіряємо, чи зображення достатньо велике
        if h < patch_size or w < patch_size:
            continue
        
        # Створюємо патчі
        for i in range(num_patches_per_image):
            # Випадкова позиція
            y = random.randint(0, h - patch_size)
            x = random.randint(0, w - patch_size)
            
            # Вирізаємо патч
            patch = img[y:y+patch_size, x:x+patch_size]
            
            # Зберігаємо патч
            patch_filename = f"{os.path.splitext(img_file)[0]}_patch_{i:03d}.png"
            patch_path = os.path.join(output_dir, patch_filename)
            cv2.imwrite(patch_path, patch)
            
            patch_count += 1
    
    print(f"Створено {patch_count} патчів в {output_dir}")

def create_low_quality_patches(hr_dir, lr_dir, patch_size=48, degradation_factor=2):
    """
    Створює низькоякісні патчі з високоякісних для тренування
    """
    os.makedirs(lr_dir, exist_ok=True)
    
    # Знаходимо всі HR патчі
    hr_files = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Обробка {len(hr_files)} HR патчів")
    
    for hr_file in tqdm(hr_files, desc="Створення LR патчів"):
        hr_path = os.path.join(hr_dir, hr_file)
        hr_img = cv2.imread(hr_path)
        
        if hr_img is None:
            continue
        
        # Створюємо низькоякісну версію
        # Зменшуємо розмір
        lr_size = (patch_size // degradation_factor, patch_size // degradation_factor)
        lr_img = cv2.resize(hr_img, lr_size, interpolation=cv2.INTER_CUBIC)
        
        # Повертаємо до оригінального розміру
        lr_img = cv2.resize(lr_img, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        
        # Додаємо шум
        noise = np.random.normal(0, 8, lr_img.shape).astype(np.uint8)
        lr_img = np.clip(lr_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Зберігаємо LR патч
        lr_path = os.path.join(lr_dir, hr_file)
        cv2.imwrite(lr_path, lr_img)

def create_demo_data():
    """
    Створює демонстраційні дані для тестування
    """
    print("Створення демонстраційних даних...")
    
    # Створюємо тестові патчі з наявних зображень
    if os.path.exists("data/hr"):
        create_test_patches("data/hr", "data/test_patches", patch_size=48, num_patches_per_image=5)
        create_low_quality_patches("data/test_patches", "data/test_patches_lr")
        print("Демонстраційні дані створені!")
    else:
        print("Папка data/hr не знайдена. Створіть її та додайте високоякісні зображення.")

if __name__ == "__main__":
    create_demo_data() 