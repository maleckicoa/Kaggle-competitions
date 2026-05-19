# Credit Risk Modelling

A binary classification project for predicting loan risk (1,000 observations).

---

## Dataset

- 1,000 observations, binary target: `Credit Risk` (0 = Good, 1 = Bad)
- Moderately imbalanced: ~70% Good / 30% Bad
- No missing values or duplicate records
- No meaningful temporal dimension
- 21 features: 15 categorical, 6 numerical

---

## Setup

- Stratified train/test split: 80/20 (800 train / 200 test)
- Model evaluation: stratified 10-fold cross-validation on the training set
- Metrics: AUC-ROC and AUC-PR (average precision)

---

## Models

| # | Model | Notes |
|---|-------|-------|
| 1 | Logistic Regression (baseline) | One-hot encoding |
| 2 | Logistic Regression | Ordinal encoding + feature engineering |
| 3 | Logistic Regression | WoE encoding |
| 4 | XGBoost | Gradient boosting |
| 5 | Random Forest | Bagging ensemble |
| 6 | Balanced Random Forest | Class-imbalance aware |
| 7 | CatBoost | Native categorical handling |

---

## Results

| Model | AUC-ROC | PR-AUC | Recall | Precision |
|-------|---------|--------|--------|-----------|
| 1 — Logistic Regression (baseline) | 0.780 | 0.606 | 0.700 | 0.503 |
| 2 — Logistic Regression (ordinal) | 0.780 | 0.590 | 0.700 | 0.520 |
| 3 — Logistic Regression (WoE) | 0.795 | 0.620 | 0.720 | 0.530 |
| 4 — XGBoost | 0.805 | 0.655 | 0.658 | 0.570 |
| 5 — Random Forest | 0.802 | 0.657 | 0.588 | 0.598 |
| 6 — Balanced Random Forest | 0.804 | 0.645 | 0.683 | 0.524 |
| 7 — CatBoost | 0.79 | 0.631 | 0.588 | 0.537 |

---

## Champion Model

A definitive champion was not selected — models 3, 4, 5, and 6 are the strongest candidates. Final selection requires defining the optimal Recall-Precision trade-off for the business context and tuning the classification threshold accordingly. Since false negatives (failing to flag a risky loan) carry significant financial consequences, recall is likely the priority metric.

---

## Remarks

- **Hyperparameter tuning:** Optuna (Bayesian optimization) was informally explored for Random Forest. For XGBoost, ad-hoc experimentation suggested the existing grid search yields comparable performance.
- **Class imbalance:** SMOTE was briefly tested but did not yield noticeable improvements.
- **KS metric:** Considered but not used — AUC-ROC and AUC-PR were preferred as they summarize performance across all thresholds rather than a single point of separation.
