# src/utils.py

import os
import json

def save_metrics(history, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training metrics to {path}")

