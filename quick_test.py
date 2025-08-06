#!/usr/bin/env python3
"""
Швидкий тест покращеного EDSR проекту
"""

import os
import sys
import torch
import cv2
import numpy as np
from models.edsr import EDSR
from config import Config

def test_model_creation():
    """Тестує створення моделі"""
    print("🔧 Тестування створення моделі...")
    
    try:
        cfg = Config()
        model = EDSR(
            upscale_factor=cfg.UPSCALE_FACTOR,
            num_blocks=cfg.NUM_BLOCKS,
            channels=cfg.CHANNELS
        )
        
        # Тестуємо forward pass
        test_input = torch.randn(1, 3, 48, 48)
        output = model(test_input)
        
        expected_size = (1, 3, 48 * cfg.UPSCALE_FACTOR, 48 * cfg.UPSCALE_FACTOR)
        
        if output.shape == expected_size:
            print("✅ Модель створена успішно")
            print(f"   Вхід: {test_input.shape}")
            print(f"   Вихід: {output.shape}")
            return True
        else:
            print(f"❌ Неправильний розмір виходу: {output.shape}, очікувалося: {expected_size}")
            return False
            
    except Exception as e:
        print(f"❌ Помилка створення моделі: {e}")
        return False

def test_data_preparation():
    """Тестує підготовку даних"""
    print("\n📊 Тестування підготовки даних...")
    
    try:
        from utils.dataset import SuperResolutionDataset
        
        # Створюємо тестове зображення
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        os.makedirs("test_data/hr", exist_ok=True)
        os.makedirs("test_data/lr", exist_ok=True)
        cv2.imwrite("test_data/hr/test.png", test_img)
        cv2.imwrite("test_data/lr/test.png", test_img)
        
        # Тестуємо датасет
        dataset = SuperResolutionDataset(
            lr_dir="test_data/lr",
            hr_dir="test_data/hr",
            patch_size=48,
            num_patches_per_image=2
        )
        
        if len(dataset) > 0:
            lr_tensor, hr_tensor = dataset[0]
            print("✅ Датасет створений успішно")
            print(f"   Розмір датасету: {len(dataset)}")
            print(f"   LR tensor shape: {lr_tensor.shape}")
            print(f"   HR tensor shape: {hr_tensor.shape}")
            
            # Очищення тестових файлів
            import shutil
            shutil.rmtree("test_data")
            return True
        else:
            print("❌ Датасет порожній")
            return False
            
    except Exception as e:
        print(f"❌ Помилка підготовки даних: {e}")
        return False

def test_patch_processing():
    """Тестує обробку патчів"""
    print("\n🔍 Тестування обробки патчів...")
    
    try:
        from predict import extract_patches, reconstruct_from_patches
        
        # Створюємо тестове зображення
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Вирізаємо патчі
        patches, positions = extract_patches(test_img, patch_size=48)
        
        if len(patches) > 0:
            # Відновлюємо зображення
            reconstructed = reconstruct_from_patches(patches, positions, test_img.shape, patch_size=48)
            
            print("✅ Обробка патчів працює")
            print(f"   Кількість патчів: {len(patches)}")
            print(f"   Розмір оригіналу: {test_img.shape}")
            print(f"   Розмір відновленого: {reconstructed.shape}")
            return True
        else:
            print("❌ Не вдалося вирізати патчі")
            return False
            
    except Exception as e:
        print(f"❌ Помилка обробки патчів: {e}")
        return False

def test_configuration():
    """Тестує конфігурацію"""
    print("\n⚙️ Тестування конфігурації...")
    
    try:
        cfg = Config()
        
        required_attrs = [
            'PATCH_SIZE', 'NUM_PATCHES_PER_IMAGE', 'UPSCALE_FACTOR',
            'NUM_BLOCKS', 'CHANNELS', 'BATCH_SIZE', 'NUM_EPOCHS',
            'LEARNING_RATE', 'DEVICE', 'SAVE_INTERVAL', 'OUTPUT_DIR'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(cfg, attr):
                missing_attrs.append(attr)
        
        if not missing_attrs:
            print("✅ Всі необхідні параметри конфігурації присутні")
            print(f"   Розмір патча: {cfg.PATCH_SIZE}")
            print(f"   Фактор збільшення: {cfg.UPSCALE_FACTOR}")
            print(f"   Пристрій: {cfg.DEVICE}")
            return True
        else:
            print(f"❌ Відсутні параметри: {missing_attrs}")
            return False
            
    except Exception as e:
        print(f"❌ Помилка конфігурації: {e}")
        return False

def main():
    """Основний тест"""
    print("🚀 Швидкий тест покращеного EDSR проекту")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_model_creation,
        test_data_preparation,
        test_patch_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Результати тестування: {passed}/{total} тестів пройдено")
    
    if passed == total:
        print("🎉 Всі тести пройдено успішно!")
        print("✅ Проект готовий до використання")
    else:
        print("⚠️ Деякі тести не пройдено")
        print("🔧 Перевірте налаштування та залежності")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 