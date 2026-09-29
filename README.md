# Customer Churn Prediction — Milestones 01–03

A supervised machine-learning project (binary classification) that predicts whether a telecom customer will churn.
Milestone 01 covers data preparation and a first baseline. Milestone 02 addresses feature engineering and class imbalance.
Milestone 03 compares two genuinely different algorithms — logistic regression and random forest.

**Headline results:**
* **Milestone 01 baseline** (logistic regression, no imbalance handling): test ROC-AUC 0.842, F1 0.604.
* **Milestone 03 model comparison** (both models class-weighted): random forest reached test F1 0.634 vs. logistic regression's 0.614, but with a noticeably larger train-test gap (0.052 vs. 0.020) — a sign of more overfitting despite the slightly higher score.

## Table of Contents
1. [Problem Definition](#1-problem-definition)
2. [Dataset](#2-dataset)
3. [Repository Structure](#3-repository-structure)
4. [Getting Started](#4-getting-started)
5. [Milestone 01 — Data Prep & Baseline](#5-milestone-01--data-prep--baseline)
6. [Milestone 02 — Feature Engineering](#6-milestone-02--feature-engineering)
7. [Milestone 03 — Model Comparison](#7-milestone-03--model-comparison)
8. [Limitations & Risks](#8-limitations--risks)
9. [Next Steps](#9-next-steps)
10. [Author & Acknowledgements](#10-author--acknowledgements)

---

## 1. Problem Definition
| Item | Definition |
| --- | --- |
| Task type | **Binary classification** |
| Target | `Churn` (`Yes`/`No` in the raw file, encoded as 1/0; churn is the positive class) |
| Business context | A telecom provider wants to find customers likely to leave, so retention offers can be targeted |
| ML objective | Estimate the probability that a customer churns from their account, service and demographic attributes |

## 2. Dataset
* **Name:** Telco Customer Churn (IBM sample dataset)
* **Source:** [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (file `WA_Fn-UseC_-Telco-Customer-Churn.csv`)
* **Rows / columns:** 7,043 rows, 21 columns
* **Churn rate:** 26.5% (1,869 of 7,043 customers)
* Not committed to this repository — see [Getting Started](#4-getting-started) to download it

## 3. Repository Structure
```text
Customer-Churn-Prediction/
├── data/
│   ├── raw/                    # place the downloaded CSV here (git-ignored)
│   └── processed/              # cleaned data written by the notebooks/scripts (git-ignored)
├── notebooks/
│   ├── milestone_01_data_prep_baseline.ipynb    # EDA, cleaning, split, pipeline, baseline, evaluation
│   ├── milestone_02_feature_engineering.ipynb   # scaling/encoding review + class-imbalance handling
│   └── milestone_03_model_comparison.ipynb      # logistic regression vs. random forest, compared and explained
├── src/
│   ├── config.py               # paths, random seed, column names
│   ├── data_preparation.py     # loading, rule-based cleaning, X/y creation
│   ├── preprocessing.py        # ColumnTransformer + baseline/class-weighted/random-forest pipelines
│   ├── evaluation.py           # metrics, cross-validation (plain + resampled), plots
│   ├── imbalance.py            # class-balance check + leakage-safe random oversampling
│   ├── train_baseline.py       # Milestone 01: end-to-end reproducible baseline run
│   ├── train_milestone02.py    # Milestone 02: compares imbalance-handling strategies
│   └── train_milestone03.py    # Milestone 03: compares logistic regression vs. random forest
├── reports/
│   ├── figures/                 # confusion matrices and ROC curves for all three milestones
│   ├── baseline_metrics.json
│   ├── milestone02_metrics.json
│   └── milestone03_metrics.json
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. Getting Started

### Setup
```bash
git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get the data
Download the CSV from the Kaggle link above and save it as `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Run everything
```bash
python -m src.train_baseline       # Milestone 01
python -m src.train_milestone02    # Milestone 02
python -m src.train_milestone03    # Milestone 03
```
Or open each notebook in order and use **Run All Cells**.

### Troubleshooting
| Problem | Fix |
| --- | --- |
| `ImportError: attempted relative import with no known parent package` | Run as a module: `python -m src.train_baseline` (not `python src/train_baseline.py`) |
| `NameError: name 'fitted' is not defined` (Milestone 03 notebook) | Run cells top to bottom, or use Run All — Section 5 depends on Section 4 having run first in the same kernel |
| `FileNotFoundError: Dataset not found` | The CSV is missing or misnamed; check `data/raw/` |

## 5. Milestone 01 — Data Prep & Baseline
EDA, rule-based cleaning (notably: 11 blank `TotalCharges` values, all belonging to customers with `tenure == 0`), a leakage-safe scikit-learn pipeline, and a plain logistic-regression baseline vs. a majority-class dummy model.

| Metric (test set) | Dummy (majority) | Logistic Regression |
| --- | ---: | ---: |
| Accuracy | 0.735 | 0.806 |
| Precision | 0.000 | 0.657 |
| Recall | 0.000 | 0.559 |
| F1-score | 0.000 | 0.604 |
| ROC-AUC | 0.500 | 0.842 |
| PR-AUC | 0.265 | 0.634 |

The baseline clearly beats the dummy model on every metric that isn't fooled by class imbalance (ROC-AUC, PR-AUC, F1), confirming there is real, learnable churn signal in the account/service features.

## 6. Milestone 02 — Feature Engineering
Reviewed the scaling (numeric) and encoding (categorical) choices from Milestone 01, confirmed the class imbalance (~27% churn in training data), and compared three strategies via cross-validation on the training set only: no adjustment, `class_weight="balanced"` logistic regression, and logistic regression on a randomly oversampled training fold (oversampling applied only within each CV fold, never across the train/validation boundary, to avoid leakage).

| Metric (test set) | Milestone 01 baseline | Milestone 02 selected strategy: |class_weight="balanced|
| --- | ---: | ---: |
| Accuracy | 0.806 | |
| Precision | 0.657 | |
| Recall | 0.559 | |
| F1-score | 0.604 | |
| ROC-AUC | 0.842 | |
| PR-AUC | 0.634 | |

## 7. Milestone 03 — Model Comparison
Two algorithms, same preprocessing, same imbalance handling (`class_weight="balanced"` / `"balanced_subsample"`), so the comparison isolates the algorithm itself:
* **Logistic regression** — linear, interpretable coefficients
* **Random forest** (300 trees, `max_depth=8`, `min_samples_leaf=5`) — tree ensemble, can capture feature interactions

### Train / cross-validation / test comparison
| | Train F1 | CV F1 (mean) | Test F1 | Train − Test gap |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.633 | 0.628 | 0.614 | 0.020 |
| Random Forest | 0.686 | 0.636 | 0.634 | 0.052 |

### Full test-set metrics
> **    "accuracy": 0.7601,
        "precision": 0.5327,
        "recall": 0.7834,
        "f1": 0.6342,
        "roc_auc": 0.842,
        "pr_auc": 0.6484  **

| Metric (test set) | Logistic Regression | Random Forest |
| --- | ---: | ---: |
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-score | 0.614 | 0.634 |
| ROC-AUC | | |
| PR-AUC | | |

### What each model considers important
Both models largely agree on the core drivers of churn — `tenure`, `TotalCharges`, `MonthlyCharges`, `Contract_Month-to-month`, `Contract_Two year`, and `InternetService_Fiber optic` all appear in both models' top-10 lists. Where they diverge, random forest also weights `OnlineSecurity_No`, `TechSupport_No`, and `PaymentMethod_Electronic check` — features that barely register for logistic regression's linear coefficients.

![Feature importance comparison](reports/figures/milestone03_feature_comparison.png)
*(Save the coefficients-vs-importances plot from notebook Section 5 to this path, or update the path to match where you saved it.)*

### Interpretation — why did the models perform differently?
Random forest reached a slightly higher test F1 (0.634 vs. 0.614), a modest ~2-point edge. The two models substantially agree on what matters most (see above), which suggests the core churn signal is strong and not an artifact of either algorithm's assumptions. Where they diverge — random forest picking up secondary signals from security/support add-ons and payment method — is consistent with churn having a real non-linear component: these features may only matter in combination with certain contract types or tenure ranges, a conditional pattern logistic regression's linear boundary can't represent directly but a tree can split on.

That said, random forest's edge comes with a cost: its train-test F1 gap (0.052) is more than double logistic regression's (0.020), and its cross-validation mean (0.636) sits much closer to its test score than to its inflated training score — a sign of overfitting that the `max_depth`/`min_samples_leaf` limits reduced but didn't eliminate. Logistic regression's near-flat train/CV/test numbers make it the more predictable, lower-variance model.

**Recommendation:** **[FILL IN — e.g., carrying forward logistic regression for its stability and interpretability, or random forest for its small performance edge, and why]**

## 8. Limitations & Risks
* Public IBM sample dataset; may not reflect real-world churn behaviour.
* Snapshot data with no timestamps — no time-based validation is possible.
* Class imbalance (26.5% churn) limits how high precision and recall can both be at a fixed threshold.
* Neither model's hyperparameters were tuned in these milestones; both use fixed, reasonable defaults.
* A single 80/20 test split gives one point estimate per model; cross-validation mitigates but doesn't eliminate this.
* Random forest's overfitting risk (Section 7) means its test-set edge over logistic regression may not fully generalize beyond this dataset.
