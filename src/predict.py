import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/best_model.h5')
class_names = ['species1', 'species2', 'species3']  # Replace with real class labels

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return class_names[np.argmax(preds)]

