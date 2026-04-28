# 🧠 ML Model Benchmark — Olivetti Faces (sklearn)

### 📋 Overview
This project tests and compares multiple **machine learning** and **deep learning** models on the **Olivetti Faces** dataset from `sklearn.datasets.fetch_olivetti_faces`.
It is designed to **evaluate model performance in terms of accuracy and training speed** for a **facial recognition task**.

> 🎓 Originally built for the CS42 *Model Selection* unit. The notebook walks through training, evaluation, side-by-side plots, and a small interactive UI for testing the best model.

---

### 🧩 Models Tested
| Category | Model | Library |
|----------|-------|---------|
| Classical ML | K-Nearest Neighbors (K=3) | scikit-learn |
| Classical ML | Decision Tree (max_depth=25) | scikit-learn |
| Ensemble ML | Random Forest (100 estimators) | scikit-learn |
| Neural Network | Multi-Layer Perceptron (MLP, 256 hidden units) | scikit-learn |
| Gradient Boosting | XGBoost | xgboost |
| Deep Learning | Convolutional Neural Network (CNN, 100 epochs) | TensorFlow / Keras |

---

### 🎯 What It Measures
| Metric | Description |
|--------|-------------|
| **Training Accuracy** | How well the model fits the training data |
| **Test Accuracy** | How well the model generalizes to unseen data |
| **Training Time (s)** | Time required to train each model |

---

### 🧠 Dataset
| Property | Details |
|----------|---------|
| **Name** | Olivetti Faces (`fetch_olivetti_faces`) |
| **Type** | 64×64 grayscale human face images |
| **Classes** | 40 subjects |
| **Samples** | 400 images total |
| **Task** | Facial recognition (multi-class classification) |

---

### 🖼 Samples & Charts

**Dataset preview:**
<img width="1175" height="269" alt="Dataset Sample Images" src="https://github.com/user-attachments/assets/5cd221b5-fc68-4ec0-b724-c05b040dbb0e" />

**MLP model predictions:**
<img width="1083" height="259" alt="MLP predictions" src="https://github.com/user-attachments/assets/d8ef8d41-3e13-4073-9334-0b6cd9e77e06" />

**Compare training accuracy:**
<img width="518" height="317" alt="Training accuracy" src="https://github.com/user-attachments/assets/a8ed0d74-8a5d-4a6b-b704-8423a0036551" />

**Compare test accuracy:**
<img width="522" height="311" alt="Test accuracy" src="https://github.com/user-attachments/assets/e6f09655-129b-4fea-bbe7-aa26f430620b" />

**Compare training time (log-scale):**
<img width="523" height="312" alt="Training time log-scale" src="https://github.com/user-attachments/assets/55b67337-0e92-4a28-b527-a9772b7a4d88" />

**Interactive UI for model testing:**
![UI Models Testing](https://github.com/user-attachments/assets/e84d65fc-d55e-47ef-ab19-1bbf9718649d)

---

### ⚙️ How to Run

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Open the notebook
jupyter notebook "ML models comparsion_v2.ipynb"
```

> 💡 Run on **Google Colab** for free GPU acceleration on the CNN — just upload the notebook to colab.google.com.

---

### 🧰 Requirements
| Package | Why |
|---------|-----|
| **Python ≥ 3.10** | Required runtime |
| `numpy`, `pandas` | Data handling |
| `matplotlib` | Plotting comparison charts |
| `scikit-learn` | Olivetti dataset + KNN, DT, RF, MLP |
| `xgboost` | Gradient boosting baseline |
| `tensorflow` | CNN training |
| `ipywidgets` | Interactive UI inside the notebook |

Install everything with `pip install -r requirements.txt`.

---

### 📁 Project Structure
```
ML-Models-Comparison---Speed-Accuracy/
├── ML models comparsion_v2.ipynb   # Main notebook (training + comparison + UI)
├── requirements.txt                # Python dependencies
├── LICENSE
└── README.md
```

---

### 🎓 Learning Objectives
1. Understand the **accuracy / speed trade-off** across model families
2. Compare **classical ML** vs **boosting** vs **deep learning** on the same data
3. Read training-curve plots and interpret signs of overfitting
4. Use `ipywidgets` to build a quick **interactive demo** for any classifier

---

### 🛑 Troubleshooting
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` (CPU build is fine for 64×64 images) |
| First-run download fails | `fetch_olivetti_faces()` downloads ~1 MB to `~/scikit_learn_data/`; check your network |
| Notebook UI widgets blank | `pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension` |

---

### 📄 License
MIT — Credits to **[CS42.org](https://cs42.org)**.
