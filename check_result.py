import cv2
import os

def check_result():
    """Перевіряє результат обробки"""
    if os.path.exists("input.jpg") and os.path.exists("result_enhanced.jpg"):
        original = cv2.imread("input.jpg")
        enhanced = cv2.imread("result_enhanced.jpg")
        
        print(f"📊 Результат обробки:")
        print(f"Оригінал: {original.shape[1]}x{original.shape[0]} пікселів")
        print(f"Покращене: {enhanced.shape[1]}x{enhanced.shape[0]} пікселів")
        print(f"Збільшення: {enhanced.shape[1]/original.shape[1]:.1f}x по ширині, {enhanced.shape[0]/original.shape[0]:.1f}x по висоті")
    else:
        print("❌ Файли не знайдено")

if __name__ == "__main__":
    check_result() 