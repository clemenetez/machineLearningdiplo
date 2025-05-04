import cv2
import os

hr_dir = 'data/train/'
lr_dir = 'data/lr/'

for img_name in os.listdir(hr_dir):
    hr_img = cv2.imread(os.path.join(hr_dir, img_name))
    lr_img = cv2.resize(hr_img, (50, 50), interpolation=cv2.INTER_CUBIC)  
    cv2.imwrite(os.path.join(lr_dir, img_name), lr_img)