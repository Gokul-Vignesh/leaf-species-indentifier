# src/config.py

import os

DATA_DIR = os.path.join(os.getcwd(), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SEGMENTED_DIR = os.path.join(DATA_DIR, 'segmented')

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

CLASS_NAMES = sorted([
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d))
])

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

