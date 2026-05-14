# Driver Drowsiness Detection with Continuous Drowsiness Index

**Course:** ARTI 560 — Computer Vision
**Approach:** Transfer Learning with MobileNetV2 on the Driver Drowsiness Dataset (DDD)

---

## 📌 Project Description

This project implements an end-to-end deep learning pipeline for **driver drowsiness detection** using a fine-tuned **MobileNetV2** convolutional neural network. The model performs binary classification (Drowsy vs. Non-Drowsy) on driver facial images and additionally produces a **continuous Drowsiness Index (DI) ∈ [0, 1]** smoothed with a moving average to reduce short-term fluctuations.

The pipeline includes dataset discovery, stratified splitting, preprocessing with light augmentation, transfer-learning model construction, training with proper callbacks, and full evaluation with exported artifacts.

---

## 🎯 Methodology

1. **Backbone:** Pretrained MobileNetV2 (ImageNet weights, frozen base).
2. **Classification head:** GlobalAveragePooling → Dropout(0.5) → Dense(64, ReLU, L2) → Dropout(0.4) → Dense(1, Sigmoid).
3. **Drowsiness Index:**

   $$DI_t = P(\text{Drowsy} \mid \text{frame}_t)$$

4. **Smoothing (moving average, window = 5):**

   $$\widetilde{DI}_t = \frac{1}{W} \sum_{i=t-W+1}^{t} DI_i$$

5. **Loss / Optimizer:** Binary Cross-Entropy with Adam (lr = 1e-4).
6. **Regularization:** Dropout, L2 weight decay, class weights for mild imbalance.

---

## 📊 Dataset

- **Name:** Driver Drowsiness Dataset (DDD)
- **Source:** [Kaggle — Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- **Total images:** 41,793
  - Non-Drowsy: 19,445 (46.53%)
  - Drowsy: 22,348 (53.47%)
- **Image format:** PNG/JPG facial frames

### Expected folder structure after download:

```
Driver Drowsiness Dataset (DDD)/
├── Drowsy/
│   ├── A0001.png
│   └── ...
└── Non Drowsy/
    ├── a0001.png
    └── ...
```

---

## 📁 Repository Structure

```
Computer-vision-Project/
├── Driver_Drowsiness_Pipeline.ipynb   # Main notebook (full pipeline)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── artifacts/                         # (generated after running)
    ├── best_mobilenetv2_drowsiness.keras
    ├── training_history.csv
    ├── drowsiness_index.csv
    ├── test_predictions_detailed.csv
    ├── evaluation_summary.json
    ├── classification_report.txt
    ├── classification_report.csv
    └── data_splits/
        ├── train_split.csv
        ├── validation_split.csv
        └── test_split.csv
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 – 3.11
- (Recommended) NVIDIA GPU with CUDA for faster training

### Setup steps

```bash
# 1. Clone the repository
git clone https://github.com/alhnoufalqhatani-create/Computer-vision-Project.git
cd Computer-vision-Project

# 2. (Optional but recommended) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1 — Download the dataset
Download the DDD dataset from [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) and extract it inside the project folder so the path looks like:

```
Computer-vision-Project/Driver Drowsiness Dataset (DDD)/Drowsy/
Computer-vision-Project/Driver Drowsiness Dataset (DDD)/Non Drowsy/
```

> The notebook auto-discovers the dataset; no path edits required as long as the folder containing `Drowsy/` and `Non Drowsy/` is inside the project root.

### Step 2 — Launch Jupyter

```bash
jupyter notebook Driver_Drowsiness_Pipeline.ipynb
```

### Step 3 — Run all cells
From the menu: **Kernel → Restart & Run All**

All outputs (model, CSVs, JSON, classification report, plots) will be written to the `artifacts/` folder automatically.

---

## 📈 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **99.74%** |
| **Test F1-Score** | **0.9976** |
| Validation Accuracy (best) | 99.71% |
| Validation Loss (best) | 0.0334 |
| Training epochs | 15 (early-stopping enabled) |

### Classification Report (Test Set)

```
              precision    recall  f1-score   support
  Non Drowsy       1.00      1.00      1.00      2917
      Drowsy       1.00      1.00      1.00      3352
    accuracy                           1.00      6269
   macro avg       1.00      1.00      1.00      6269
weighted avg       1.00      1.00      1.00      6269
```

---

## 🔁 Reproducibility

- A global random seed (`SEED = 42`) is set across Python, NumPy, and TensorFlow.
- Train/validation/test splits are exported as CSVs in `artifacts/data_splits/` so the exact same partition can be reused.
- The best-performing model checkpoint is saved as `best_mobilenetv2_drowsiness.keras`.

---

## 🧪 Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 8 |
| Initial epochs | 15 |
| Learning rate | 1e-4 |
| Smoothing window | 5 |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Train / Val / Test split | 70% / 15% / 15% |

---

## 👤 Author

**Alhnouf Alqhatani**
ARTI 560 — Computer Vision Project

---

## 📄 License

This project is submitted for academic purposes as part of ARTI 560 coursework.
