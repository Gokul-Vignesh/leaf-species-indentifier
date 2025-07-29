import cv2
import os

def resize_images(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, folder)
        save_path = os.path.join(output_dir, folder)
        os.makedirs(save_path, exist_ok=True)

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)
                cv2.imwrite(os.path.join(save_path, img_file), img)

