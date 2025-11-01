# 🧠 ML Model Benchmark — Olivetti Faces (sklearn)

## 📋 Overview
This project tests and compares multiple **machine learning** and **deep learning models** on the **Olivetti Faces dataset** from `sklearn.datasets.fetch_olivetti_faces`.  
It is designed to **evaluate model performance in terms of accuracy and training speed** for a **facial recognition task**.

---

## 🧩 Models Tested
| Category | Model | Library |
|-----------|--------|----------|
| Classical ML | K-Nearest Neighbors (K=3) | scikit-learn |
| Classical ML | Decision Tree (max_depth=25) | scikit-learn |
| Ensemble ML | Random Forest (200 estimators) | scikit-learn |
| Neural Network | Multi-Layer Perceptron (MLP, 256 hidden units) | scikit-learn |
| Gradient Boosting *(optional)* | XGBoost | xgboost |
| Deep Learning *(optional)* | Convolutional Neural Network (CNN) | TensorFlow / Keras |

---

## 🎯 What It Measures
| Metric | Description |
|---------|--------------|
| **Training Accuracy** | How well the model fits the training data |
| **Test Accuracy** | How well the model generalizes to unseen data |
| **Training Time (s)** | Time required to train each model |

---

## 🧠 Dataset
| Property | Details |
|-----------|----------|
| **Name** | Olivetti Faces (`fetch_olivetti_faces`) |
| **Type** | 64×64 grayscale human face images |
| **Classes** | 40 subjects |
| **Samples** | 400 images total |
| **Task** | Facial recognition (multi-class classification) |

---

## ⚙️ How to Run
```bash
# 1. Create and activate a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) enable extras
# pip install xgboost tensorflow

# 4. Run the notebook
jupyter notebook
