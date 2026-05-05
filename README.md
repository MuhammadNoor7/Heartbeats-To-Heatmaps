# DS-3002 Data Mining — Assignment #4 · Spring 2026

**Course:** DS-3002 Data Mining | **Institution:** FAST-NUCES · BSDS | **Student:** Muhammad Noor (i232520)

> **Heartbeat to Heatmap:** Unsupervised Learning, Ensemble Methods, and Neural Networks on Heart Disease & Handwritten Digit Data

---

## Project Overview

An end-to-end data mining pipeline covering:
- **Unsupervised Learning** — K-Means, Hierarchical Clustering, PCA, t-SNE on UCI Heart Disease data
- **Ensemble Methods** — Random Forest (Bagging) and XGBoost (Boosting) with SHAP interpretability
- **Neural Networks** — SLP, MLP (with ablation study), and a lightweight CNN for MNIST digit recognition
- **Interactive Dashboard** — Local Streamlit app for real-time heart disease risk prediction

---

## Repository Structure

```
assignment4/
├── notebooks/          # Jupyter notebooks with full analysis
├── app/                # Streamlit dashboard application
│   ├── app.py
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── requirements.txt
├── report/             # LaTeX source and final PDF report
├── heart+disease/      # UCI Heart Disease dataset (Cleveland subset)
├── screenshots/        # App screenshots for report submission
└── requirements.txt    # Root-level dependencies for the notebook
```

---

## Dataset Download Steps

### Dataset 1 — UCI Heart Disease (Cleveland)
Used in Parts Pre, A, B, C, E.

**Option A (already included):** The dataset is pre-loaded in the `heart+disease/` folder as `processed.cleveland.data`.

**Option B (manual download):**
1. Visit the UCI ML Repository: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
2. Download the dataset ZIP and extract `processed.cleveland.data`.
3. Place it inside the `heart+disease/` folder at the root of this repository.

### Dataset 2 — MNIST Handwritten Digits
Used in Part D (CNN only).

**No download required.** MNIST is loaded automatically inside the notebook using:
```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
Only the first 12,000 training images and 2,000 test images are used (array slicing applied in the notebook).

---

## How to Run the Notebook

1. **Install dependencies** from the root of the repository:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the notebook:**
   Navigate to the `notebooks/` folder and open:
   ```
   i232520_MuhammadNoor_A4_DataMining.ipynb
   ```

4. **Run all cells:**
   - Go to **Kernel → Restart & Run All**
   - All random seeds are fixed (`random_state=42`) to ensure fully reproducible results
   - Estimated total runtime: **< 15 minutes** on a standard laptop (CPU only, no GPU needed)

---

## How to Run the App

The interactive Streamlit dashboard allows real-time heart disease risk prediction using the trained best model.

1. **Navigate to the app folder:**
   ```bash
   cd app
   ```

2. **Install app dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. The app opens automatically in your browser at `http://localhost:8501`.
   - Fill in the 13 patient feature fields (a sample patient is pre-populated for instant testing).
   - Click **Predict** to see the risk label, confidence score, and top SHAP feature drivers.

---

## Requirements

All dependencies for the notebook are listed in [`requirements.txt`](requirements.txt) at the root. Install with:
```bash
pip install -r requirements.txt
```

Key libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `streamlit`, `shap`, `matplotlib`, `seaborn`, `scipy`, `joblib`, `imbalanced-learn`

