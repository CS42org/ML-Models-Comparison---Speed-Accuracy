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
| Ensemble ML | Random Forest (100 estimators) | scikit-learn |
| Neural Network | Multi-Layer Perceptron (MLP, 256 hidden units) | scikit-learn |
| Gradient Boosting  | XGBoost | xgboost |
| Deep Learning  | Convolutional Neural Network (CNN) (100 Epochs) | TensorFlow / Keras |

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
<img width="1083" height="259" alt="image" src="https://github.com/user-attachments/assets/d8ef8d41-3e13-4073-9334-0b6cd9e77e06" />

## Compare Training Accuracy
<img width="518" height="317" alt="image" src="https://github.com/user-attachments/assets/a8ed0d74-8a5d-4a6b-b704-8423a0036551" />

## Compare Test Accuracy
<img width="522" height="311" alt="image" src="https://github.com/user-attachments/assets/e6f09655-129b-4fea-bbe7-aa26f430620b" />

## Compare Training Time (s) Log-scale
<img width="523" height="312" alt="image" src="https://github.com/user-attachments/assets/55b67337-0e92-4a28-b527-a9772b7a4d88" />


## UI Model Testing

![UI Models Testing ](https://github.com/user-attachments/assets/e84d65fc-d55e-47ef-ab19-1bbf9718649d)



## ⚙️ How to Run
```bash
# 1. Create and activate a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook
