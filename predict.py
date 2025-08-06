import torch
import cv2
import numpy as np
from models.edsr import EDSR  
from config import Config
import os
from tqdm import tqdm

def extract_patches(image, patch_size=48, overlap=8):
    """Вирізає патчі з зображення з перекриттям"""
    patches = []
    positions = []
    
    h, w = image.shape[:2]
    
    # Якщо зображення менше за патч, використовуємо його цілком
    if h < patch_size or w < patch_size:
        # Збільшуємо зображення до розміру патча
        resized_image = cv2.resize(image, (patch_size, patch_size))
        patches.append(resized_image)
        positions.append((0, 0))
        return patches, positions
    
    # Для малих зображень використовуємо менший overlap
    if h < patch_size * 2 or w < patch_size * 2:
        overlap = 4
    
    for y in range(0, h - patch_size + 1, patch_size - overlap):
        for x in range(0, w - patch_size + 1, patch_size - overlap):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch)
                positions.append((x, y))
    
    # Якщо не вдалося вирізати патчі, використовуємо все зображення
    if len(patches) == 0:
        resized_image = cv2.resize(image, (patch_size, patch_size))
        patches.append(resized_image)
        positions.append((0, 0))
    
    return patches, positions

def reconstruct_from_patches(patches, positions, original_shape, patch_size=48, upscale_factor=2, overlap=8):
    """Відновлює зображення з патчів"""
    h, w = original_shape[:2]
    
    # Якщо був тільки один патч, повертаємо його
    if len(patches) == 1:
        return cv2.resize(patches[0], (w * upscale_factor, h * upscale_factor))
    
    # Визначаємо розмір покращеного патча
    enhanced_patch_size = patch_size * upscale_factor
    
    # Розраховуємо розмір результуючого зображення
    result_h = h * upscale_factor
    result_w = w * upscale_factor
    
    result = np.zeros((result_h, result_w, 3), dtype=np.float32)
    count = np.zeros((result_h, result_w), dtype=np.float32)
    
    for patch, (x, y) in zip(patches, positions):
        # Масштабуємо позицію для результуючого зображення
        result_x = x * upscale_factor
        result_y = y * upscale_factor
        
        # Перевіряємо межі
        if (result_y + enhanced_patch_size <= result_h and 
            result_x + enhanced_patch_size <= result_w):
            result[result_y:result_y+enhanced_patch_size, 
                   result_x:result_x+enhanced_patch_size] += patch
            count[result_y:result_y+enhanced_patch_size, 
                  result_x:result_x+enhanced_patch_size] += 1
    
    # Усереднюємо перекриваючі області
    count[count == 0] = 1  # Уникаємо ділення на нуль
    result = result / count[..., np.newaxis]
    
    return result.astype(np.uint8)

def upscale_image_patches(model, lr_image_path, output_path, device, patch_size=48, upscale_factor=2):
    """Покращує зображення, обробляючи його по патчах"""
    
    # Завантажуємо зображення
    lr_img = cv2.imread(lr_image_path)
    if lr_img is None:
        raise FileNotFoundError(f"Файл {lr_image_path} не знайдено!")
    
    original_shape = lr_img.shape
    print(f"Оригінальний розмір: {original_shape}")
    
    # Вирізаємо патчі
    patches, positions = extract_patches(lr_img, patch_size)
    print(f"Вирізано {len(patches)} патчів")
    
    if len(patches) == 0:
        print("Не вдалося вирізати патчі. Спробуйте зменшити розмір патча.")
        return
    
    # Обробляємо кожен патч
    enhanced_patches = []
    
    with torch.no_grad():
        for patch in tqdm(patches, desc="Обробка патчів"):
            # Підготовка патча
            patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)
            patch_tensor = (patch_tensor / 127.5) - 1.0
            patch_tensor = patch_tensor.unsqueeze(0).to(device)
            
            # Передбачення
            enhanced_tensor = model(patch_tensor)
            
            # Конвертація назад
            enhanced_tensor = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_patch = (enhanced_tensor * 127.5 + 127.5).astype(np.uint8)
            
            enhanced_patches.append(enhanced_patch)
    
    # Відновлюємо зображення
    enhanced_image = reconstruct_from_patches(enhanced_patches, positions, original_shape, patch_size, upscale_factor)
    
    # Зберігаємо результат
    cv2.imwrite(output_path, enhanced_image)
    print(f"Покращене зображення збережено: {output_path}")
    
    return enhanced_image

def upscale_image_simple(model, lr_image_path, output_path, device):
    """Простий метод покращення всього зображення (для порівняння)"""
    
    lr_img = cv2.imread(lr_image_path)
    if lr_img is None:
        raise FileNotFoundError(f"Файл {lr_image_path} не знайдено!")

    # Підготовка зображення
    lr_tensor = torch.tensor(lr_img, dtype=torch.float32).permute(2, 0, 1)  
    lr_tensor = (lr_tensor / 127.5) - 1.0  
    lr_tensor = lr_tensor.unsqueeze(0).to(device)  

    # Передбачення
    with torch.no_grad():
        hr_tensor = model(lr_tensor)

    # Конвертація результату
    hr_tensor = (hr_tensor.squeeze().permute(1, 2, 0).cpu().numpy())  
    hr_image = (hr_tensor * 127.5 + 127.5).astype(np.uint8)  

    # Збереження
    cv2.imwrite(output_path, hr_image)
    print(f"Зображення збережено: {output_path}")

def create_comparison(original_path, enhanced_path, output_path):
    """Створює порівняння оригінального та покращеного зображень"""
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)
    
    if original is None or enhanced is None:
        print("Помилка завантаження зображень для порівняння")
        return
    
    # Зменшуємо розмір для кращого відображення
    height = min(original.shape[0], enhanced.shape[0], 400)
    width = min(original.shape[1], enhanced.shape[1], 600)
    
    original_resized = cv2.resize(original, (width, height))
    enhanced_resized = cv2.resize(enhanced, (width, height))
    
    # Створюємо порівняння
    comparison = np.hstack([original_resized, enhanced_resized])
    
    # Додаємо підписи
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Enhanced', (width + 10, 30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"Порівняння збережено: {output_path}")

if __name__ == "__main__":
    cfg = Config()
    
    # Шляхи до файлів
    model_path = "results/best_model.pth"  # Використовуємо найкращу модель
    if not os.path.exists(model_path):
        model_path = "results/final_model.pth"  # Fallback до фінальної моделі
    
    input_image = "input.jpg"  # Змініть на вашу фотографію
    output_image = "result_enhanced.jpg"
    comparison_image = "comparison.jpg"

    # Перевірка наявності файлів
    if not os.path.exists(model_path):
        print(f"Помилка: Модель не знайдена в {model_path}")
        print("Спочатку потрібно натренувати модель!")
        exit(1)
    
    if not os.path.exists(input_image):
        print(f"Помилка: Зображення не знайдено {input_image}")
        print("Помістіть вашу фотографію в корінь проекту та змініть назву в скрипті")
        exit(1)

    # Завантаження моделі з правильними параметрами для існуючої моделі
    model = EDSR(
        upscale_factor=cfg.UPSCALE_FACTOR,
        num_blocks=cfg.NUM_BLOCKS,
        channels=cfg.CHANNELS
    ).to(cfg.DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()
    print(f"Модель завантажена з {model_path}")
    print(f"Параметри: UPSCALE_FACTOR={cfg.UPSCALE_FACTOR}, NUM_BLOCKS={cfg.NUM_BLOCKS}, CHANNELS={cfg.CHANNELS}")

    # Покращення зображення
    try:
        enhanced_img = upscale_image_patches(
            model, 
            input_image, 
            output_image, 
            cfg.DEVICE, 
            patch_size=cfg.PATCH_SIZE,
            upscale_factor=cfg.UPSCALE_FACTOR
        )
        
        # Створюємо порівняння
        create_comparison(input_image, output_image, comparison_image)
        
        print("✅ Обробка завершена успішно!")
        print(f"📁 Результати:")
        print(f"  - {output_image}: Покращена фотографія (збільшена в 10 разів)")
        print(f"  - {comparison_image}: Порівняння")
        
    except Exception as e:
        print(f"❌ Помилка при обробці: {e}")
        print("💡 Спробуйте:")
        print("  1. Перевірити назву файлу зображення")
        print("  2. Натренувати нову модель: python train.py")
        print("  3. Зменшити розмір патча в config.py")