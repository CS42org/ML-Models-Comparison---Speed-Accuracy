# Olivetti Face Recognition Model Comparison

This document summarizes the workflow implemented in `ML models comparsion_v2.ipynb`, which compares multiple machine-learning models on the Olivetti faces dataset.

## Dataset and Preprocessing

* The notebook downloads the Olivetti faces dataset via `fetch_olivetti_faces`, yielding 64×64 grayscale headshots and 40 identity labels.【F:ML models comparsion_v2.ipynb†L82-L86】
* Images are split into training and testing subsets with a 40/60 split, stratified by identity and controlled by a `random_state` of 42 for reproducibility.【F:ML models comparsion_v2.ipynb†L96-L99】
* Classical models receive flattened, standardized pixel intensities, while the convolutional network retains image tensors with a channel dimension appended.【F:ML models comparsion_v2.ipynb†L101-L111】

## Models in the Benchmark

The helper `train_and_report` routine fits a model, reports training and testing accuracy along with training time, and visualizes sample predictions for qualitative inspection.【F:ML models comparsion_v2.ipynb†L123-L136】 The notebook benchmarks the following learners:

| Model | Key Parameters |
|-------|----------------|
| K-Nearest Neighbors | 3 neighbors and shared data scaler.【F:ML models comparsion_v2.ipynb†L164-L166】 |
| Decision Tree | Maximum depth of 25 and a fixed random seed.【F:ML models comparsion_v2.ipynb†L194-L196】 |
| Random Forest | 200 estimators trained in parallel with `n_jobs=-1`.【F:ML models comparsion_v2.ipynb†L224-L226】 |
| Multilayer Perceptron | Single hidden layer of 256 units, trained for up to 200 iterations.【F:ML models comparsion_v2.ipynb†L254-L256】 |
| XGBoost (optional) | Gradient-boosted trees (requires the `xgboost` package).【F:ML models comparsion_v2.ipynb†L284-L287】 |
| Convolutional Neural Network (optional) | Two convolutional blocks followed by dense layers; activated when TensorFlow is available.【F:ML models comparsion_v2.ipynb†L319-L347】 |

## Interactive Exploration

After training, an `ipywidgets` interface exposes the fitted models so you can select an estimator and preview its predictions on hold-out images.【F:ML models comparsion_v2.ipynb†L363-L378】 This UI leverages the cached test tensors stored in `TEST_CACHE` during preprocessing.【F:ML models comparsion_v2.ipynb†L101-L111】

## Running the Notebook

1. Install Python 3.9+ with the required packages: `numpy`, `matplotlib`, `scikit-learn`, and `ipywidgets`. Optional comparisons additionally use `xgboost` and `tensorflow`.
2. Launch Jupyter Lab or Notebook and open `ML models comparsion_v2.ipynb`.
3. Execute the cells sequentially to train the models, visualize qualitative predictions, and interact with the widget-based explorer.

Because the notebook measures wall-clock training time, you may observe different timings depending on CPU, GPU availability, and whether optional libraries are installed.
