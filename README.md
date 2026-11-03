# Credit Card Fraud Detection

This project is an end-to-end machine learning pipeline for **detecting fraudulent credit card transactions**. The main challenge is **severe class imbalance** (fewer than 0.2% of transactions are fraud). Four tree-based models are trained and tuned, compared on a validation set, and the best one (LightGBM) is evaluated on a held-out test set and saved together with its tuned decision threshold.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models and Hyperparameter Tuning](#models-and-hyperparameter-tuning)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Using the Saved Model](#using-the-saved-model)
- [File Structure](#file-structure)
- [Notes and Limitations](#notes-and-limitations)

---

## Dataset

The data is loaded from [OpenML](https://www.openml.org/) (dataset ID `1597`) via `fetch_openml`. It is the well-known Credit Card Fraud dataset.

| Property | Value |
|---|---|
| Initial number of records | 284,807 |
| Columns | 30 (29 features + `Class` label) |
| Features | `V1`–`V28` (anonymized components) and `Amount` (transaction amount) |
| Label | `Class`: `0` = normal, `1` = fraud |
| Missing values | None |

**Class distribution (before cleaning):**

| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 284,315 | 99.8273% |
| Fraud (1) | 492 | 0.1727% |

---

## Project Workflow

### 1. Data Cleaning
- 9,144 **duplicate rows** were detected and removed (the duplicated rows included 14,261 normal and 32 fraud records).
- Size after deduplication: **275,663 rows**
  - Normal: 275,190
  - Fraud: 473

### 2. Stratified Data Split
Splits preserve class proportions (`stratify`) with `random_state=42`:

| Set | Samples | Normal | Fraud | Purpose |
|---|---|---|---|---|
| Fit | 176,424 | 176,122 | 302 | Training and hyperparameter search (with CV) |
| Validation | 44,106 | 44,030 | 76 | Model comparison and threshold selection |
| Test | 55,133 | 55,038 | 95 | Final evaluation (used only once) |

First, 20% of the data is set aside for testing; then 20% of the remaining 80% is held out for validation.

### 3. Model Training and Tuning
All four models were tuned on the Fit set using `average_precision` (equivalent to PR-AUC) with 3-fold cross-validation. Average Precision was chosen over Accuracy or ROC-AUC because it is a more reliable metric for highly imbalanced data.

### 4. Decision Threshold Tuning
For each model, thresholds from `0.1` to `0.9` (step 0.1) were evaluated on the **Validation** set, and the threshold maximizing **F1** was selected.

### 5. Model Comparison
Models were ranked by Average Precision on the Validation set, and their Precision–Recall curves were plotted.

### 6. Final Training and Evaluation
The winning model (LightGBM) was retrained with the best hyperparameters on the **entire training data** (Fit + Validation) and evaluated exactly once on the Test set.

### 7. Persistence
The final model and the selected threshold are saved with `joblib`.

---

## Models and Hyperparameter Tuning

| Model | Search method | Imbalance handling | Best hyperparameters | CV Average Precision |
|---|---|---|---|---|
| **XGBoost** | `GridSearchCV` (27 combinations) | None (GPU: `device="cuda"`, `tree_method="hist"`) | `learning_rate=0.1`, `max_depth=7`, `n_estimators=300` | 0.8562 |
| **Random Forest** | `RandomizedSearchCV` (10 iterations) | `class_weight="balanced"` | `n_estimators=200`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features="sqrt"` | 0.8238 |
| **LightGBM** | `RandomizedSearchCV` (20 iterations) | `class_weight="balanced"` | `n_estimators=500`, `learning_rate=0.1`, `num_leaves=15`, `max_depth=8`, `min_child_samples=30`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=0.1` | 0.8601 |
| **CatBoost** | `RandomizedSearchCV` (10 iterations) | `auto_class_weights="Balanced"` | `iterations=500`, `depth=8`, `learning_rate=0.05`, `l2_leaf_reg=7` | 0.8478 |

---

## Results

### Model Comparison on the Validation Set

| Model | Best threshold | Average Precision | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **LightGBM** | 0.6 | **0.7985** | 0.9375 | 0.7895 | **0.8571** |
| CatBoost | 0.9 | 0.7982 | 0.8429 | 0.7763 | 0.8082 |
| Random Forest | 0.7 | 0.7808 | 0.8382 | 0.7500 | 0.7917 |
| XGBoost | 0.4 | 0.7770 | 0.9516 | 0.7763 | 0.8551 |

LightGBM achieved the highest Average Precision and the highest F1, so it was selected as the final model.

### Final Evaluation on the Test Set (LightGBM, threshold = 0.6)

| Metric (fraud class) | Value |
|---|---|
| Precision | 0.88 |
| Recall | 0.79 |
| F1-score | 0.83 |
| **Average Precision** | **0.8483** |

**Confusion Matrix:**

|  | Predicted: Normal | Predicted: Fraud |
|---|---|---|
| **Actual: Normal** | 55,028 (TN) | 10 (FP) |
| **Actual: Fraud** | 20 (FN) | 75 (TP) |

Of the 95 fraudulent transactions in the test set, 75 were detected, and only 10 normal transactions were incorrectly flagged as fraud.

---

## Installation and Usage

### Requirements
- Python 3.9 or higher
- (Optional) A CUDA-compatible GPU for XGBoost; otherwise change `device="cuda"` to `device="cpu"`.

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost joblib jupyter
```

### Run

```bash
jupyter notebook main.ipynb
```

Run the cells in order. Downloading the data from OpenML requires an internet connection.

> **Note:** The hyperparameter searches (especially CatBoost and LightGBM) may take several minutes.

---

## Using the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load("final_lightgbm_model.pkl")
threshold = joblib.load("classification_threshold.pkl")

# new_data must contain columns V1..V28 and Amount, in the same order
proba = model.predict_proba(new_data)[:, 1]
is_fraud = (proba >= threshold).astype(int)
```

---

## File Structure

```
.
├── main.ipynb                       # Main project notebook
├── final_lightgbm_model.pkl         # Final LightGBM model (created after running)
├── classification_threshold.pkl     # Decision threshold (0.6)
└── README.md
```

---

## Notes and Limitations

- **Few fraud samples:** The test set contains only 95 fraud cases, so reported metrics (e.g., Recall) have considerable statistical variance, and small differences between models are not necessarily significant.
- **The optimal threshold is F1-based.** In real-world applications, the threshold should be tuned according to the business cost of errors (cost of a missed fraud vs. cost of blocking a legitimate transaction).
- **No resampling:** Imbalance is handled via class weighting (`class_weight` / `auto_class_weights`); SMOTE or similar techniques were not used. XGBoost was trained without class weighting.
- **Deduplication before splitting** prevents identical samples from appearing in both training and test sets (avoiding data leakage).
- The features `V1`–`V28` are anonymized and have no direct business interpretation.
- The final model is saved as a `pickle` file; only load `.pkl` files from trusted sources, and keep library versions in the deployment environment consistent with the training environment.
