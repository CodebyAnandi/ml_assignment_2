"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates them, and saves:
    - trained model objects (model/*.pkl)
    - the fitted StandardScaler (model/scaler.pkl)
    - the list of feature names (model/feature_names.pkl)
    - test_data.csv (the held-out test split, used by the Streamlit app)
    - metrics_summary.csv (the comparison table used in README.md)

Dataset: sklearn.datasets.load_breast_cancer
    - 569 instances (> 500 required)
    - 30 numeric features (> 12 required)
    - Binary classification: malignant (0) vs benign (1)
    This is the same dataset publicly hosted on UCI ML Repository /
    Kaggle as "Breast Cancer Wisconsin (Diagnostic) Data Set".
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class balance:\n{y.value_counts()}\n")

# ---------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only, then transform both)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. Save test data (raw, unscaled features + true label) for the app
# ---------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(ROOT / "test_data.csv", index=False)
print(f"Saved test_data.csv with {test_df.shape[0]} rows")

# ---------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(name, metrics)

    # Save trained model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(HERE / f"{fname}.pkl", "wb") as f:
        pickle.dump(model, f)

# ---------------------------------------------------------------------
# 6. Save scaler + feature names (needed by the app to preprocess uploads)
# ---------------------------------------------------------------------
with open(HERE / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(HERE / "feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# ---------------------------------------------------------------------
# 7. Save metrics summary (used to build the README comparison table)
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(ROOT / "metrics_summary.csv", index=False)
print("\nFinal comparison table:\n")
print(results_df.to_string(index=False))
