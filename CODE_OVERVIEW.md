# ML Models Comparison – Speed vs Accuracy

This document summarizes the workflow implemented in [`ML models comparsion_v2.ipynb`](ML%20models%20comparsion_v2.ipynb), which benchmarks classical and deep-learning classifiers on the Olivetti Faces dataset.

![Highlighted `train_and_report` helper](assets/train_and_report.svg)

## Dataset preparation
* Loads the shuffled Olivetti Faces dataset (64×64 grayscale portraits) and shows a gallery of sample images to contextualize the classification task.【F:ML models comparsion_v2.ipynb†L83-L90】
* Splits the data into training and test sets with stratification, flattens images for classical models, and standardizes pixel values using `StandardScaler`. Separate cached views are kept for flat and convolutional inputs.【F:ML models comparsion_v2.ipynb†L97-L110】

## Shared training/reporting helper
* The `train_and_report` function times model fitting, calculates train/test accuracy, plots example predictions, and stores trained estimators in a registry for later interactive use.【F:ML models comparsion_v2.ipynb†L123-L140】

## Model zoo
* **KNN, Decision Tree, Random Forest, MLP:** Trained on standardized flattened pixels to compare traditional algorithms across complexity/speed trade-offs.【F:ML models comparsion_v2.ipynb†L165-L256】
* **XGBoost:** Optionally trained when the library is available, offering gradient-boosted trees for higher accuracy at extra cost.【F:ML models comparsion_v2.ipynb†L285-L298】
* **CNN:** When TensorFlow is installed, a compact convolutional network is trained directly on image tensors, delivering end-to-end learning and softmax outputs.【F:ML models comparsion_v2.ipynb†L318-L343】

## Interactive exploration
* An `ipywidgets` UI lets notebook users pick any trained model and inspect up to 20 prediction overlays, aiding qualitative comparison beyond numeric metrics.【F:ML models comparsion_v2.ipynb†L363-L381】

## Takeaways
* The notebook offers a reproducible framework for contrasting diverse classifiers on a consistent face-recognition benchmark, emphasizing the trade-offs between setup complexity, training time, and predictive accuracy.【F:ML models comparsion_v2.ipynb†L83-L381】
