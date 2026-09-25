# Credit Risk Scoring

A walkthrough of decision trees and ensemble methods applied to credit risk scoring — predicting whether a loan applicant will default, using the classic `CreditScoring.csv` dataset.

## What's inside

- **Data cleaning & prep** — decoding categorical fields (`status`, `home`, `marital`, `records`, `job`), fixing sentinel missing values (`99999999` → `NaN`), and building train/validation/test splits.
- **Decision Trees** — training a baseline tree, diagnosing overfitting, and tuning `max_depth` / `min_samples_leaf`.
- **Random Forest** — tuning `n_estimators`, `max_depth`, and `min_samples_leaf` via AUC curves.
- **XGBoost** — training, monitoring, and tuning `eta`, `max_depth`, and `min_child_weight`.
- **Model selection** — comparing decision tree, random forest, and XGBoost on validation AUC, then evaluating the final XGBoost model on the held-out test set.
- **Export** — saving the trained model to `credit_risk_model.pkl` with `joblib`.

## Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
joblib
```

## Usage

1. Place `CreditScoring.csv` in the project directory (source: [mlbookcamp-code](https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv)).
2. Run `Credit_Risk_Scoring.ipynb` top to bottom.
3. The final model is saved as `credit_risk_model.pkl`.

## Result

The tuned XGBoost model (`eta=0.3`, `max_depth=4`, `min_child_weight=1`, `num_boost_round=35`) was selected as the best performer and evaluated for final AUC on the test set.
