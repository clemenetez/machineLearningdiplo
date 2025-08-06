import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.edsr import EDSR
from config import Config
from predict import upscale_image_patches, upscale_image_simple, create_comparison
import os

def demonstrate_patch_vs_traditional():
    """
    Демонструє різницю між традиційним та патч-базованим підходами
    """
    cfg = Config()
    
    # Шляхи до файлів
    model_path = "results/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "results/final_model.pth"
    
    input_image = "face.png"
    
    if not os.path.exists(input_image):
        print(f"Тестове зображення {input_image} не знайдено!")
        return
    
    # Завантаження моделі
    model = EDSR(
        upscale_factor=cfg.UPSCALE_FACTOR,
        num_blocks=cfg.NUM_BLOCKS,
        channels=cfg.CHANNELS
    ).to(cfg.DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        print(f"Модель завантажена з {model_path}")
    else:
        print("Модель не знайдена. Спочатку потрібно натренувати модель!")
        return
    
    model.eval()
    
    print("=== Демонстрація різниці підходів ===")
    print("1. Традиційний підхід: стиснення всього зображення")
    print("2. Патч-базований підхід: вирізання та відновлення патчів")
    
    # Традиційний підхід
    print("\n--- Традиційний підхід ---")
    try:
        upscale_image_simple(model, input_image, "result_traditional.jpg", cfg.DEVICE)
        print("✓ Традиційний підхід завершено")
    except Exception as e:
        print(f"✗ Помилка в традиційному підході: {e}")
    
    # Патч-базований підхід
    print("\n--- Патч-базований підхід ---")
    try:
        upscale_image_patches(model, input_image, "result_patches.jpg", cfg.DEVICE, cfg.PATCH_SIZE)
        print("✓ Патч-базований підхід завершено")
    except Exception as e:
        print(f"✗ Помилка в патч-базованому підході: {e}")
    
    # Створення порівняння
    print("\n--- Створення порівняння ---")
    try:
        create_comparison("result_traditional.jpg", "result_patches.jpg", "comparison_methods.jpg")
        print("✓ Порівняння створено")
    except Exception as e:
        print(f"✗ Помилка створення порівняння: {e}")
    
    print("\n=== Результати збережено ===")
    print("- result_traditional.jpg: Традиційний підхід")
    print("- result_patches.jpg: Патч-базований підхід")
    print("- comparison_methods.jpg: Порівняння методів")

def analyze_image_quality(image_path):
    """
    Аналізує якість зображення та виводить статистику
    """
    if not os.path.exists(image_path):
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Базова статистика
    h, w = img.shape[:2]
    mean_brightness = np.mean(img)
    std_brightness = np.std(img)
    
    # Розрахунок різкості (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'size': (w, h),
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'sharpness': laplacian_var
    }

def compare_results():
    """
    Порівнює результати різних методів
    """
    print("\n=== Аналіз якості результатів ===")
    
    files_to_analyze = [
        ("Оригінал", "face.png"),
        ("Традиційний підхід", "result_traditional.jpg"),
        ("Патч-базований підхід", "result_patches.jpg")
    ]
    
    results = {}
    
    for name, file_path in files_to_analyze:
        stats = analyze_image_quality(file_path)
        if stats:
            results[name] = stats
            print(f"\n{name}:")
            print(f"  Розмір: {stats['size']}")
            print(f"  Середня яскравість: {stats['mean_brightness']:.1f}")
            print(f"  Стандартне відхилення: {stats['std_brightness']:.1f}")
            print(f"  Різкість: {stats['sharpness']:.2f}")
        else:
            print(f"\n{name}: Файл не знайдено або помилка завантаження")
    
    return results

def create_visual_comparison():
    """
    Створює візуальне порівняння всіх результатів
    """
    print("\n=== Створення візуального порівняння ===")
    
    images = []
    titles = []
    
    for title, path in [("Оригінал", "face.png"), 
                       ("Традиційний", "result_traditional.jpg"),
                       ("Патч-базований", "result_patches.jpg")]:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # Конвертуємо BGR в RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Зменшуємо розмір для відображення
                img_resized = cv2.resize(img_rgb, (300, 300))
                images.append(img_resized)
                titles.append(title)
    
    if len(images) >= 2:
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("visual_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Візуальне порівняння збережено як 'visual_comparison.png'")
    else:
        print("✗ Недостатньо зображень для порівняння")

if __name__ == "__main__":
    print("🚀 Демонстрація покращеного EDSR проекту")
    print("=" * 50)
    
    # Основна демонстрація
    demonstrate_patch_vs_traditional()
    
    # Аналіз результатів
    compare_results()
    
    # Візуальне порівняння
    create_visual_comparison()
    
    print("\n" + "=" * 50)
    print("✅ Демонстрація завершена!")
    print("\n📁 Перевірте створені файли:")
    print("- result_traditional.jpg")
    print("- result_patches.jpg") 
    print("- comparison_methods.jpg")
    print("- visual_comparison.png") 