import cv2
import os
from collections import Counter

def check_lr_sizes():
    lr_dir = 'data/lr'
    sizes = []
    
    for filename in os.listdir(lr_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(lr_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                sizes.append((width, height))
    
    # Підраховуємо унікальні розміри
    size_counts = Counter(sizes)
    
    print("Розміри LR зображень:")
    for size, count in size_counts.most_common():
        print(f"{size[0]}x{size[1]}: {count} зображень")
    
    print(f"\nВсього зображень: {len(sizes)}")
    
    # Перевіряємо чи всі розміри однакові
    if len(size_counts) == 1:
        print("✅ Всі зображення мають однаковий розмір")
    else:
        print("❌ Зображення мають різні розміри - це може бути проблемою!")
        
        # Показуємо перші кілька прикладів
        print("\nПриклади різних розмірів:")
        for i, (size, count) in enumerate(size_counts.most_common(5)):
            print(f"  {size[0]}x{size[1]}: {count} зображень")

if __name__ == "__main__":
    check_lr_sizes() 