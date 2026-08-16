"""
train_models.py
----------------
Trains 5 classification models on the "Default of Credit Card Clients"
dataset (UCI Machine Learning Repository) to predict whether a client will
default on their credit card payment next month.

Dataset source: UCI ML Repository
    "Default of Credit Card Clients Dataset"
    https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    (Yeh, I. C., & Lien, C. H., 2009)

    - 30,000 instances (>> 500 required)
    - 23 features (>> 12 required)
    - Binary target: default.payment.next.month (0 = no default, 1 = default)

Outputs:
    - trained model objects (model/*.pkl)
    - fitted StandardScaler (model/scaler.pkl)
    - feature name list (model/feature_names.pkl)
    - test_data.csv (held-out test split for the Streamlit app)
    - metrics_summary.csv (single train/test split metrics)
    - cv_summary.csv (5-fold cross-validation mean +/- std, for robustness)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
# 1. Load and clean the dataset
# ---------------------------------------------------------------------
raw = pd.read_csv(ROOT / "raw_data.csv")

# Drop the ID column - not a predictive feature, just a row identifier
raw = raw.drop(columns=["ID"])

# Rename target to something simpler
raw = raw.rename(columns={"default.payment.next.month": "target"})

X = raw.drop(columns=["target"])
y = raw["target"]

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class balance:\n{y.value_counts()}\n")

# ---------------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. Save a manageable test_data.csv for the Streamlit app
#    (Streamlit Community Cloud free tier has limited resources, so we
#    keep the full 6000-row test split - still well within free-tier limits
#    for a CSV upload widget, but noted here in case a smaller sample is
#    preferred for faster demo loads.)
# ---------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(ROOT / "test_data.csv", index=False)
print(f"Saved test_data.csv with {test_df.shape[0]} rows")

# ---------------------------------------------------------------------
# 5. Define models
#    Decision Tree is depth-constrained (max_depth=6) to prevent the
#    severe overfitting an unconstrained tree shows on this dataset.
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = []
cv_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    # --- single-split metrics (used for the app + main comparison table) ---
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

    # --- 5-fold cross-validation on the full (scaled) dataset for robustness ---
    X_full_scaled = scaler.transform(X)  # reuse scaler fit on training split
    cv_f1 = cross_val_score(model, X_full_scaled, y, cv=cv, scoring="f1", n_jobs=-1)
    cv_results.append({
        "ML Model Name": name,
        "CV F1 Mean": round(cv_f1.mean(), 4),
        "CV F1 Std": round(cv_f1.std(), 4),
    })

    # Save trained model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(HERE / f"{fname}.pkl", "wb") as f:
        pickle.dump(model, f)

# ---------------------------------------------------------------------
# 6. Save scaler + feature names
# ---------------------------------------------------------------------
with open(HERE / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(HERE / "feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# ---------------------------------------------------------------------
# 7. Save summaries
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(ROOT / "metrics_summary.csv", index=False)

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(ROOT / "cv_summary.csv", index=False)

print("\nFinal comparison table (single 80/20 split):\n")
print(results_df.to_string(index=False))
print("\n5-fold cross-validation F1 (mean +/- std):\n")
print(cv_df.to_string(index=False))
