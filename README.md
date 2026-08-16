# ML Assignment 2 — Breast Cancer Classification with Streamlit

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple binary
classification models that predict whether a breast tumor is **malignant**
or **benign** based on numeric features computed from a digitized image of a
fine needle aspirate (FNA) of a breast mass. Five classification algorithms
are trained on the same dataset, compared using standard evaluation metrics,
and demonstrated through an interactive Streamlit web application.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set
**Source:** UCI Machine Learning Repository / also available on Kaggle
(https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
accessed here via `sklearn.datasets.load_breast_cancer` for reproducibility.

- **Instances:** 569 (≥ 500 required)
- **Features:** 30 numeric features (≥ 12 required) — computed from cell
  nuclei characteristics such as radius, texture, perimeter, area,
  smoothness, compactness, concavity, symmetry, and fractal dimension
  (mean, standard error, and "worst" value for each of 10 base measurements).
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign

The held-out test split (114 rows, 20% of the data) is saved as
`test_data.csv` in this repository and is the file used to demonstrate the
Streamlit app.

## c. GitHub Repository Link

> **TODO:** Replace this with your actual GitHub repo link after you push
> this project, e.g. `https://github.com/<your-username>/ml-assignment-2`

## d. Models Used

All 5 models were trained on the same 80/20 train-test split
(`random_state=42`, stratified) with features standardized using
`StandardScaler` (fit on training data only).

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|:--------:|:------:|:---------:|:------:|:------:|:------:|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                         | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                 | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)    | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

*(Exact values are reproducible by running `model/train_models.py`; they are
also saved to `metrics_summary.csv`.)*

### Observations

| ML Model Name             | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression        | Best overall performer on this dataset. Since the standardized features separate the two classes fairly linearly, a linear decision boundary generalizes very well, giving the highest accuracy, F1, and MCC of all five models. |
| Decision Tree               | Weakest performer. A single unpruned tree overfits the training data and does not generalize as well, producing the lowest accuracy, AUC, and MCC — the more noticeable drop in recall suggests it misses more actual malignant/benign cases at the decision boundary. |
| kNN                          | Strong performer once features are scaled (scaling is essential for distance-based models). Performs identically to Random Forest on accuracy/F1 here, showing the classes are fairly well separated in feature space. |
| Naive Bayes                  | Reasonable performance despite its strong (and technically incorrect) assumption of feature independence — the 30 features are actually correlated (e.g., radius, perimeter, area). Its AUC is surprisingly close to the top models even though accuracy is lower, meaning it ranks predictions well but its default 0.5 threshold isn't optimal. |
| Random Forest (Ensemble)    | Second-best model overall. Averaging many decision trees reduces the overfitting seen in the single Decision Tree, recovering most of the performance lost by that model and nearly matching Logistic Regression, with the second-highest AUC. |
| **Overall Winner for this dataset** | **Logistic Regression** — highest Accuracy (0.9825), F1 (0.9861), and MCC (0.9623), indicating both strong class separation by a linear boundary and balanced performance across malignant/benign classes. |

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split used by the app
│-- metrics_summary.csv     # metrics for all 5 models (source for table above)
│-- model/
│   │-- train_models.py     # trains all 5 models, saves .pkl files
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl          # fitted StandardScaler
│   │-- feature_names.pkl   # list of the 30 feature column names
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models, test_data.csv, metrics_summary.csv
streamlit run app.py
```

## How to Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (include the `model/*.pkl` files).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app** → select this repository and branch (`main`).
4. Set the main file path to `app.py`.
5. Click **Deploy**.
6. Once live, upload `test_data.csv` in the app to see model results.

## Live Streamlit App Link

> **TODO:** Replace with your deployed app URL, e.g.
> `https://<your-app-name>.streamlit.app`
