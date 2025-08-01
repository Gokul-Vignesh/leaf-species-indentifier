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
