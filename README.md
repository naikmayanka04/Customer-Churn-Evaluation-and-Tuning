# Customer Churn Prediction — Milestone 01: Data Prep & Baseline

Supervised machine-learning project (binary classification) that predicts whether a telecom customer will churn.
This milestone covers data exploration, cleaning, a leakage-safe preprocessing pipeline, a simple baseline model, and its evaluation.
Hyperparameter tuning and more complex models are intentionally left for later milestones.

## Problem
* **Task:** binary classification — `Churn` (Yes/No) → encoded as 1/0, churn is the positive class
* **Business context:** identify customers likely to leave so retention offers can be targeted
* **Baseline model:** Logistic Regression, compared against a majority-class `DummyClassifier`

## Dataset
* **Name:** Telco Customer Churn (IBM sample dataset)
* **Where to get it:** search Kaggle for "Telco Customer Churn" and download `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
  Check the dataset page for its license and terms of use.
* **Where to put it:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
* The raw data is **not committed** to the repository (see `.gitignore`); download it yourself using the steps above.
* Use the standard 21-column CSV. Some expanded versions contain columns such as `Churn Label`, `Churn Score` or `Churn Reason`
  that leak the target; `src/data_preparation.py` drops them if present, but the standard file does not need this.

## Repository structure
```text
churn-project/
├── data/
│   ├── raw/                    # place the downloaded CSV here (git-ignored)
│   └── processed/              # cleaned data written by the notebook/script (git-ignored)
├── notebooks/
│   ├── milestone_01_data_prep_baseline.ipynb    # EDA, cleaning, split, pipeline, baseline, evaluation
│   ├── milestone_02_feature_engineering.ipynb   # scaling/encoding review + class-imbalance handling
│   └── milestone_03_model_comparison.ipynb      # logistic regression vs. random forest, compared and explained
├── 04_evaluation_tuning.ipynb   # classification_report/ROC/confusion matrix + GridSearchCV tuning (repo root, not notebooks/)
├── src/
│   ├── config.py               # paths, random seed, column names
│   ├── data_preparation.py     # load, rule-based cleaning, X/y creation
│   ├── preprocessing.py        # ColumnTransformer + baseline/class-weighted/random-forest pipelines
│   ├── evaluation.py           # metrics, cross-validation (plain + resampled), plots
│   ├── imbalance.py            # class-balance check + leakage-safe random oversampling
│   ├── tuning.py                # GridSearchCV parameter grids and builder, per model type
│   ├── train_baseline.py       # Milestone 01: end-to-end reproducible baseline run
│   ├── train_milestone02.py    # Milestone 02: compares imbalance-handling strategies, evaluates the best one
│   ├── train_milestone03.py    # Milestone 03: compares logistic regression vs. random forest
│   └── train_milestone04.py    # Milestone 04: cross-validation, GridSearchCV tuning, rigorous final evaluation
├── reports/
│   ├── figures/                # confusion matrix, ROC curve (created by the script)
│   └── baseline_metrics.json   # metrics from the last script run
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to reproduce
1. Place the CSV in `data/raw/` (see above).
2. **Notebooks (full analysis):** run `jupyter lab`, then open and run top to bottom, in order:
   `notebooks/milestone_01_data_prep_baseline.ipynb`, `notebooks/milestone_02_feature_engineering.ipynb`,
   `notebooks/milestone_03_model_comparison.ipynb`, and `04_evaluation_tuning.ipynb` (this one sits at the repository root,
   not inside `notebooks/`). Each notebook reloads and re-splits the data itself (same seed), so any of them can also be run
   on its own.
3. **Scripts:** from the repository root:
   ```bash
   python -m src.train_baseline       # Milestone 01 baseline
   python -m src.train_milestone02    # Milestone 02: imbalance-handling comparison + final evaluation
   python -m src.train_milestone03    # Milestone 03: logistic regression vs. random forest
   python -m src.train_milestone04    # Milestone 04: cross-validation, GridSearchCV tuning, final evaluation
   ```
   Each writes its own metrics file and figures under `reports/`, so no milestone overwrites another's results.

## Method summary
| Stage | Approach |
| --- | --- |
| Cleaning | Rule-based only: whitespace stripping, `TotalCharges` to numeric, deterministic handling of blank values (see notebook Section 5), exact-duplicate removal |
| Split | 80/20, stratified on the target, `random_state=42`, done before any fitting |
| Preprocessing | `ColumnTransformer`: numeric → median impute + standard scale; categorical → most-frequent impute + one-hot |
| Leakage control | All learned transformations live inside a scikit-learn `Pipeline` fitted on training data only; cross-validation refits them per fold |
| Model selection | None in this milestone; the test set is evaluated once |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix (accuracy alone is misleading under class imbalance) |

## Results
> Fill this table from **your own run** (`reports/baseline_metrics.json` or the notebook output). Do not copy numbers from elsewhere.

| Metric (test set) | Dummy (majority) | Logistic Regression |
| --- | ---: | ---: |
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-score | | |
| ROC-AUC | | |
| PR-AUC | | |

Short interpretation (2–4 sentences, written from your results):

## Limitations
* Public sample dataset; may not reflect real-world churn behaviour.
* Snapshot data without timestamps, so no time-based validation.
* Class imbalance; a single 80/20 split gives one point estimate.
* Untuned linear baseline at the default 0.5 decision threshold.

## Milestone 02 — Feature Engineering
Builds on Milestone 01's cleaning and preprocessing. Adds:
* A short review of the existing scaling (numeric) and encoding (categorical) choices against the actual feature distributions.
* **Class-imbalance detection**: churners are the minority class in the training split (about 26–32% depending on split, see the notebook).
* Three strategies compared with cross-validation **on the training set only**: no adjustment (Milestone 01 baseline), `class_weight="balanced"`
  logistic regression, and logistic regression trained on a randomly oversampled training fold (oversampling implemented in `src/imbalance.py`,
  applied strictly inside each CV fold and never to validation/test data, to avoid leaking duplicated rows).
* The strategy with the best training-CV F1 is fitted once on the full training set and evaluated once on the test set.

> Fill in from your own run of `python -m src.train_milestone02` (or `reports/milestone02_metrics.json`):

| Metric (test set) | Milestone 01 baseline | Milestone 02 selected strategy: _____ |
| --- | ---: | ---: |
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-score | | |
| ROC-AUC | | |
| PR-AUC | | |

Short interpretation (does recall improve over Milestone 01, and at what cost to precision?):

## Next steps after Milestone 02
Threshold selection based on retention cost, regularisation tuning, feature engineering
(e.g. tenure bands, service-count features), probability calibration, and more robust evaluation (repeated CV, confidence intervals).
Model comparison (logistic regression vs. random forest) is taken up next, in Milestone 03 below.

## Milestone 03 — Model Comparison
Trains and compares two genuinely different algorithms on identical preprocessing and imbalance handling, so any difference in
results reflects the algorithm itself:
* **Logistic regression** (`class_weight="balanced"`) — a linear model with signed, additive, interpretable coefficients.
* **Random forest** (`class_weight="balanced_subsample"`, `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`) — a tree
  ensemble that can capture feature interactions automatically, at the cost of interpretability and a higher risk of overfitting.

Both are compared with cross-validation on the training set, then fitted once and evaluated once on the test set. Training-set
scores are also recorded to check each model's train/test gap for signs of overfitting.

> Fill in from your own run of `python -m src.train_milestone03` (or `reports/milestone03_metrics.json`):

| Metric (test set) | Logistic Regression | Random Forest |
| --- | ---: | ---: |
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-score | | |
| ROC-AUC | | |
| PR-AUC | | |
| Train − test F1 gap | | |

**Why they performed differently on this data** (fill in from notebook Section 7 — this is the milestone's mentor-review question):

## Next steps (later milestones)
Hyperparameter tuning for whichever model is carried forward, gradient boosting as a third comparison point, threshold selection
based on retention cost, probability calibration, and more robust evaluation (repeated CV, confidence intervals).

## Milestone 04 — Evaluation & Tuning
Evaluates both class-weighted models with metrics appropriate for an imbalanced classification problem — `classification_report`,
`confusion_matrix`, `roc_auc_score`, and a manually plotted ROC curve — instead of accuracy alone. Uses `cross_val_score` directly
to establish untuned baselines (scored on ROC-AUC), then runs `GridSearchCV` separately for each model type with its own parameter
grid (5-fold stratified CV, scored on F1):

| Model | Parameters searched |
| --- | --- |
| Logistic Regression | `C`: 0.01, 0.1, 1, 10, 100 |
| Random Forest | `n_estimators`: 100/300/500; `max_depth`: 4/8/12/None; `min_samples_leaf`: 1/5/10 |

The tuned model with the higher cross-validated F1 is selected and evaluated once on the held-out test set. The notebook also
programmatically generates a plain-language explanation of precision, recall, F1 and ROC-AUC using the actual numbers from that
run, so the write-up always matches whatever data it was run on.

> Fill in from your own run of `python -m src.train_milestone04` (or `reports/milestone04_metrics.json`):

| Metric (test set, tuned model) | Value |
| --- | ---: |
| Selected model | |
| Best hyperparameters | |
| Precision (churn) | |
| Recall (churn) | |
| F1-score (churn) | |
| ROC-AUC | |

## Next steps (later milestones)
Threshold selection based on retention-offer cost (using the ROC curve's threshold array), probability calibration, a wider or
randomised hyperparameter search, gradient boosting as a further comparison point, and more robust evaluation (repeated CV,
confidence intervals).

