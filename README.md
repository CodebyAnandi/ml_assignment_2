# ML Assignment 2 — Credit Card Default Prediction with Streamlit

## a. Problem Statement

Credit card issuers need to identify clients likely to default on their
payments so they can manage risk proactively. This assignment builds and
compares five classification models that predict whether a credit card
client will **default on payment next month** (`target = 1`) or not
(`target = 0`), based on their credit profile and repayment history. The
best-performing model is demonstrated through an interactive Streamlit
web application.

## b. Dataset Description

**Dataset:** Default of Credit Card Clients Dataset
**Source:** UCI Machine Learning Repository (Yeh, I. C. & Lien, C. H., 2009)
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

- **Instances:** 30,000 (≥ 500 required)
- **Features:** 23 numeric features (≥ 12 required), including:
  - `LIMIT_BAL` — amount of credit given
  - `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` — demographics
  - `PAY_0` … `PAY_6` — repayment status over 6 months
  - `BILL_AMT1` … `BILL_AMT6` — bill statement amounts over 6 months
  - `PAY_AMT1` … `PAY_AMT6` — previous payment amounts over 6 months
- **Target:** Binary — `1 = defaulted next month`, `0 = did not default`
- **Class balance:** 23,364 no-default / 6,636 default (~22% default rate —
  **this dataset is meaningfully imbalanced**, unlike a roughly 50/50 split)
- **Missing values:** none

The held-out test split (6,000 rows, 20% of the data) is saved as
`test_data.csv` and is the file used to demonstrate the Streamlit app.

## c. GitHub Repository Link

> **TODO:** (https://github.com/CodebyAnandi/ml_assignment_2)
> `https://github.com/CodebyAnandi/ml_assignment_2`

## d. Models Used

All 5 models were trained on the same 80/20 stratified train-test split
(`random_state=42`) with features standardized using `StandardScaler`
(fit on training data only, to avoid leakage). To check that results
weren't just an artifact of one lucky split, 5-fold stratified
cross-validation (scored on F1, since accuracy is a poor metric on
imbalanced data) was also run on the full dataset — see the CV table
below the main comparison table.

**Hyperparameter tuning process:** the Decision Tree was tested at two
depths, `max_depth=6` and `max_depth=4` (both with `min_samples_leaf=20`).
Random Forest was tested at `n_estimators=300` and `n_estimators=500`
(both with `max_depth=10`). The final values used below are `max_depth=4`
for Decision Tree and `n_estimators=500` for Random Forest — see the
tuning note under the tables for what was observed and why these values
were chosen.

### Comparison Table (held-out 20% test split, 6,000 rows)

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------------|:--------:|:------:|:---------:|:------:|:------:|:------:|
| Logistic Regression        | 0.8077   | 0.7076 | 0.6868    | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree               | 0.8187   | 0.7360 | 0.6626    | 0.3670 | 0.4724 | 0.3974 |
| kNN                         | 0.8063   | 0.7327 | 0.6147    | 0.3331 | 0.4321 | 0.3499 |
| Naive Bayes                 | 0.7525   | 0.7249 | 0.4515    | 0.5539 | 0.4975 | 0.3386 |
| Random Forest (Ensemble)    | 0.8177   | 0.7746 | 0.6667    | 0.3512 | 0.4600 | 0.3898 |

### 5-Fold Cross-Validation (F1 score, full dataset)

| ML Model Name             | CV F1 Mean | CV F1 Std |
|-----------------------------|:----------:|:---------:|
| Logistic Regression        | 0.3599     | 0.0068    |
| Decision Tree               | 0.4602     | 0.0107    |
| kNN                         | 0.4361     | 0.0060    |
| Naive Bayes                 | 0.4953     | 0.0204    |
| Random Forest (Ensemble)    | 0.4660     | 0.0059    |

*(All values reproducible by running `model/train_models.py`; also saved
to `metrics_summary.csv` and `cv_summary.csv`.)*

**Hyperparameter tuning note:** the Decision Tree was tested at
`max_depth=6` and `max_depth=4`. Contrary to the initial expectation that
a shallower tree would underfit, `max_depth=4` actually performed *better*
on every metric (F1 rose from 0.4492 to 0.4724, MCC from 0.3825 to 0.3974)
and was substantially more stable across cross-validation folds (std
dropped from 0.0215 to 0.0107). This suggests `max_depth=6` still had some
residual overfitting despite already being constrained — `max_depth=4`
sits closer to the actual bias-variance sweet spot for this dataset.
Random Forest was also tested at `n_estimators=300` vs `500`; performance
was essentially unchanged (MCC 0.3908 → 0.3898), showing 300 trees was
already enough for the ensemble average to stabilize — more trees only
added compute cost with no real benefit.

### Observations

| ML Model Name             | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression        | Highest accuracy-looking result at a glance (80.8%), but this is misleading: recall is only 0.24, meaning it misses roughly 3 out of 4 actual defaulters. A linear boundary struggles to separate the classes here, and the model defaults to predicting "no default" for most borderline cases — a direct consequence of the class imbalance (78% of clients don't default). |
| Decision Tree               | Strongest tree-based result after tuning `max_depth=4` — now has the **highest MCC (0.3974) of all five models**, and the most stable cross-validation performance among the non-linear models (std 0.0107). Captures non-linear repayment-history patterns (e.g., specific `PAY_0` thresholds) that Logistic Regression misses, while staying simple enough to avoid the overfitting seen at deeper settings. |
| kNN                          | Middling performance across the board. Distance-based similarity works reasonably on scaled features, but doesn't clearly outperform simpler models here, suggesting the decision boundary isn't primarily about local neighborhoods in feature space. |
| Naive Bayes                  | Lowest accuracy (75.3%) but **highest F1 (0.4975) and highest recall (0.55)** of all five models — it catches more actual defaulters than any other model, at the cost of more false alarms (lower precision, 0.45). Its independence assumption is technically wrong here (bill amounts across months are highly correlated), but this apparently biases it toward flagging risk more readily, which is arguably desirable in a credit-risk context where missing a defaulter is costlier than a false alarm. |
| Random Forest (Ensemble)    | Best on AUC (0.7746) and most stable overall (lowest CV std, 0.0059), but after tuning, no longer the top MCC — the tuned Decision Tree now edges it out (0.3974 vs 0.3898), suggesting the extra complexity of averaging 500 trees isn't buying much over a single well-tuned tree on this dataset. |
| **Overall Winner for this dataset** | **Depends on the business objective.** By **MCC**, the tuned **Decision Tree** is now the strongest single model (0.3974). By **AUC and ranking stability**, **Random Forest** is still the most reliable general-purpose choice. By **recall** — arguably the most important metric for credit risk, since missing a defaulter is costlier than a false alarm — **Naive Bayes** is clearly the best, catching 55% of actual defaulters versus roughly a third for the other non-linear models. This is a case where accuracy alone would give a misleading picture, and "best model" genuinely depends on what the bank is optimizing for. |

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split used by the app (6,000 rows)
│-- raw_data.csv            # original downloaded dataset
│-- metrics_summary.csv     # single-split metrics for all 5 models
│-- cv_summary.csv          # 5-fold cross-validation F1 scores
│-- model/
│   │-- train_models.py     # trains all 5 models, saves .pkl files
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl          # fitted StandardScaler
│   │-- feature_names.pkl   # list of the 23 feature column names
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models, test_data.csv, metrics
streamlit run app.py
```

## How to Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (include the `model/*.pkl` files).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app** → select this repository and branch (`main`).
4. Set the main file path to `app.py`.
5. Click **Deploy**.
6. Once live, upload `test_data.csv` in the app's sidebar to see model results.

## Live Streamlit App Link

> **TODO:** Replace with your deployed app URL, e.g.
> `https://<your-app-name>.streamlit.app`

## Limitations

- This is historical data from Taiwan (2005) — repayment behavior and
  credit norms may not generalize to other regions or time periods.
- Class imbalance (~22% default rate) was addressed only through metric
  choice (F1, MCC over raw accuracy) and hyperparameter tuning; techniques
  like class weighting or SMOTE resampling were not applied and could
  likely improve recall further, especially for Logistic Regression.
- A single 80/20 split was used for the main comparison table; the 5-fold
  CV results show some variation (particularly for Naive Bayes), so
  reported metrics should be read as estimates, not exact values.
- Hyperparameter tuning here was manual (testing 2 values per parameter),
  not an exhaustive grid search — a full `GridSearchCV` sweep could
  potentially find better settings than what was tested.