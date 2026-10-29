# Iris Classification Benchmark

A reproducible, end-to-end classification benchmark using the built-in Iris dataset. It compares Logistic Regression and Decision Tree models, reports held-out classification metrics, and creates exploratory and model-interpretation plots.

## Requirements

Python 3.10 or later and the packages in [`requirements.txt`](./requirements.txt).

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python iris_classification.py
```

The default run uses a stratified 80/20 train/test split and random seed `42`. Dataset normalization is fitted inside each model pipeline on training data only. The Decision Tree is scaled for a consistent pipeline; scaling is not required for tree-based models.

```powershell
python iris_classification.py --output-dir outputs --test-size 0.2 --random-state 42
```

## Outputs

The script creates the output directory and saves:

- `iris_dataset.csv` — the Iris features and species labels.
- `metrics.csv` — test accuracy, weighted precision, weighted recall, and weighted F1 for each model.
- `iris_pairplot.png` — Seaborn feature-pair exploration.
- `confusion_matrices.png` — test-set confusion matrices for both models.
- `decision_boundaries.png` — illustrative two-feature decision boundaries. These visualization-only models use the first two features; benchmark metrics are calculated from models trained on all four features.

Per-class precision, recall, and F1 reports are also printed to the console. The contact-form and portfolio-site task is not part of this machine-learning exercise.
