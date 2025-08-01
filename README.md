# Plant Species Identifier

A college-level plant species identification project using leaf images (Leafsnap dataset).

## 🔍 Overview

- Train a CNN (MobileNetV2) to classify leaf images into species.
- Flask backend serves predictions via a `/predict` API.
- Simple frontend for users to upload an image and get predicted species.

## 📁 Project Structure

plant-species-identifier/
├── data/
│ ├── raw/
│ ├── processed/
│ └── segmented/
├── notebooks/
│ └── EDA.ipynb
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
├── app/
│ ├── api.py
│ └── model_loader.py
├── frontend/
│ ├── index.html
│ ├── styles.css
│ └── script.js
├── outputs/
│ ├── models/
│ ├── logs/
│ └── results/
├── main.py
├── requirements.txt
└── README.md

🚀 Setup & Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Preprocess data (optional) & train:

bash
Copy
Edit
python main.py train
Evaluate model:

bash
Copy
Edit
python main.py eval
Serve backend:

bash
Copy
Edit
./run_app.sh
Serve frontend:

bash
Copy
Edit
cd frontend && python3 -m http.server 8000
📚 Files Description
src/: core scripts for training, evaluation, and data loading.

app/: Flask backend serving prediction API.

frontend/: HTML/JS UI for user uploads.

main.py: CLI entrypoint.
