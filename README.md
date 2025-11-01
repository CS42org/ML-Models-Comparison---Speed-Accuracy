# 🧠 ML Model Benchmark — Olivetti Faces (sklearn)

## 📋 Overview
This project benchmarks multiple **machine learning models** and a **simple CNN** on the **Olivetti Faces dataset** from `sklearn.datasets.fetch_olivetti_faces`.  
It compares models in terms of **training accuracy**, **testing accuracy**, and **training time**.  
A small interactive UI is included to visualize model predictions on random face samples.

---

## 🧩 Models Compared
| Type | Model |
|------|--------|
| Classical ML | K-Nearest Neighbors (K=3), Decision Tree, Random Forest, MLP |
| Gradient Boosting *(optional)* | XGBoost |
| Deep Learning *(optional)* | CNN (Keras/TensorFlow) |

---

## 🎯 What It Measures
| Metric | Description |
|---------|--------------|
| **Training Accuracy** | Accuracy on training data |
| **Test Accuracy** | Accuracy on unseen test data |
| **Training Time (s)** | Model fit time in seconds |

---

## 🧠 Dataset
| Property | Details |
|-----------|----------|
| **Name** | Olivetti Faces (via `fetch_olivetti_faces`) |
| **Type** | 64×64 grayscale human faces |
| **Classes** | 40 subjects (multi-class classification) |
| **Samples** | 400 total images |
| **Task** | Face recognition benchmark |

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
