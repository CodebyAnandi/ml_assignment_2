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

> **TODO:** Replace with your actual GitHub repo link, e.g.
> `https://github.com/<your-username>/credit-card-default-classifier`

## d. Models Used

All 5 models were trained on the same 80/20 stratified train-test split
(`random_state=42`) with features standardized using `StandardScaler`
(fit on training data only, to avoid leakage). To check that results
weren't just an artifact of one lucky split, 5-fold stratified
cross-validation (scored on F1, since accuracy is a poor metric on
imbalanced data) was also run on the full dataset — see the CV table
below the main comparison table.

**Note on hyperparameters:** the Decision Tree was constrained
(`max_depth=6, min_samples_leaf=20`) rather than left fully unconstrained.
An unconstrained tree on this dataset overfits badly (near-perfect
training accuracy but much worse generalization) — constraining depth
trades a little training fit for meaningfully better generalization.

### Comparison Table (held-out 20% test split, 6,000 rows)

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------------|:--------:|:------:|:---------:|:------:|:------:|:------:|
| Logistic Regression        | 0.8077   | 0.7076 | 0.6868    | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree               | 0.8165   | 0.7479 | 0.6682    | 0.3384 | 0.4492 | 0.3825 |
| kNN                         | 0.8065   | 0.7327 | 0.6153    | 0.3338 | 0.4328 | 0.3507 |
| Naive Bayes                 | 0.7525   | 0.7249 | 0.4515    | 0.5539 | 0.4975 | 0.3386 |
| Random Forest (Ensemble)    | 0.8180   | 0.7740 | 0.6686    | 0.3512 | 0.4605 | 0.3908 |

### 5-Fold Cross-Validation (F1 score, full dataset)

| ML Model Name             | CV F1 Mean | CV F1 Std |
|-----------------------------|:----------:|:---------:|
| Logistic Regression        | 0.3599     | 0.0068    |
| Decision Tree               | 0.4540     | 0.0215    |
| kNN                         | 0.4361     | 0.0059    |
| Naive Bayes                 | 0.4953     | 0.0204    |
| Random Forest (Ensemble)    | 0.4672     | 0.0046    |

*(All values reproducible by running `model/train_models.py`; also saved
to `metrics_summary.csv` and `cv_summary.csv`.)*

### Observations

| ML Model Name             | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression        | Highest accuracy-looking result at a glance (80.8%), but this is misleading: recall is only 0.24, meaning it misses roughly 3 out of 4 actual defaulters. A linear boundary struggles to separate the classes here, and the model defaults to predicting "no default" for most borderline cases — a direct consequence of the class imbalance (78% of clients don't default). |
| Decision Tree               | Once depth-constrained, performs solidly and stably (low CV std of 0.0215 relative to its mean). Captures non-linear repayment-history patterns (e.g., specific `PAY_0` thresholds) that a linear model misses, roughly doubling recall over Logistic Regression. |
| kNN                          | Middling performance across the board. Distance-based similarity works reasonably on scaled features, but doesn't clearly outperform simpler models here, suggesting the decision boundary isn't primarily about local neighborhoods in feature space. |
| Naive Bayes                  | Lowest accuracy (75.3%) but **highest F1 (0.4975) and highest recall (0.55)** of all five models — it catches more actual defaulters than any other model, at the cost of more false alarms (lower precision, 0.45). Its independence assumption is technically wrong here (bill amounts across months are highly correlated), but this apparently biases it toward flagging risk more readily, which is arguably desirable in a credit-risk context where missing a defaulter is costlier than a false alarm. |
| Random Forest (Ensemble)    | Best on Accuracy, AUC (0.774), and MCC (0.3908) — the most balanced performer overall, benefiting from averaging many trees to reduce the variance a single Decision Tree shows. Also has the most stable cross-validation score (std of only 0.0046). |
| **Overall Winner for this dataset** | **Depends on the business objective.** By MCC/AUC/overall balance, **Random Forest** is the strongest general-purpose model. However, if the priority is catching as many actual defaulters as possible (a common real-world priority in credit risk, since missed defaults are costly), **Naive Bayes** — with its far higher recall — is arguably the better practical choice despite a lower accuracy. This is a case where accuracy alone would give a misleading picture of the "best" model. |

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
  choice (F1, MCC over raw accuracy); techniques like class weighting or
  SMOTE resampling were not applied and could likely improve recall further.
- A single 80/20 split was used for the main comparison table; the 5-fold
  CV results show some variation (particularly for Decision Tree and Naive
  Bayes), so reported metrics should be read as estimates, not exact values.
