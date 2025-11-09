# 🧠 ML Model Benchmark — Olivetti Faces (sklearn)
#Note: CNN Model require modifications - accuracy should be much higher

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
| Gradient Boosting  | XGBoost | xgboost |
| Deep Learning  | Convolutional Neural Network (CNN) *Model need configuration* | TensorFlow / Keras |

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
## Samples

## Dataset Sample Images
<img width="1175" height="269" alt="Dataset Sample Images" src="https://github.com/user-attachments/assets/5cd221b5-fc68-4ec0-b724-c05b040dbb0e" />

## MLP Model Predicitons
<img width="1170" height="269" alt="MLP Model Prediction Sample" src="https://github.com/user-attachments/assets/8d9d020c-9117-4cad-8c5a-aee8601ab56c" />

## Compare Training Accuracy
<img width="790" height="390" alt="ML Models Train Acc" src="https://github.com/user-attachments/assets/76199024-30ca-4bed-81e2-9712d000c4f4" />

## Compare Test Accuracy
<img width="790" height="390" alt="ML Models Test Acc" src="https://github.com/user-attachments/assets/7ee34cd0-45bc-4326-a73e-df17e6505e0d" />

## Compare Time (s)
<img width="790" height="390" alt="ML Model Times" src="https://github.com/user-attachments/assets/72e8be9a-2fba-45fe-a239-e2359081b7e7" />


## UI Model Testing

![UI Models Testing ](https://github.com/user-attachments/assets/e84d65fc-d55e-47ef-ab19-1bbf9718649d)



## ⚙️ How to Run
```bash
# 1. Create and activate a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

#Note: CNN Model require modifications - accuracy should be much higher

# 3. (Optional) enable extras
# pip install xgboost tensorflow

# 4. Run the notebook
jupyter notebook
