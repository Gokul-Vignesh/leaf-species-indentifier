# create_class_names.py
import os

data_dir = 'data/raw'
output_path = 'outputs/class_names.txt'

classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
os.makedirs('outputs', exist_ok=True)

with open(output_path, 'w') as f:
    for class_name in classes:
        f.write(f"{class_name}\n")

print(f"✅ Saved {len(classes)} class names to {output_path}")
