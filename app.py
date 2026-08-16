"""
Streamlit App — Breast Cancer Classification Demo
----------------------------------------------------
Upload the provided test_data.csv (or any CSV with the same 30 feature
columns + a 'target' column) and compare 5 trained classification models.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))
    feature_names = pickle.load(open(MODEL_DIR / "feature_names.pkl", "rb"))
    models = {}
    for name, fname in MODEL_FILES.items():
        with open(MODEL_DIR / fname, "rb") as f:
            models[name] = pickle.load(f)
    return scaler, feature_names, models


scaler, feature_names, models = load_artifacts()

st.title("🔬 Breast Cancer Classification — Model Demo")
st.markdown(
    """
    This app demonstrates **5 classification models** trained on the
    [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
    (569 instances, 30 features, binary classification: malignant vs benign).

    Upload the test CSV below, pick a model, and view its performance.
    """
)

# ---------------------------------------------------------------------
# 1. Dataset upload
# ---------------------------------------------------------------------
st.header("1️⃣ Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload test_data.csv (must contain the 30 feature columns + a 'target' column)",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column with the true labels.")
        st.stop()

    X = df[feature_names]
    y_true = df["target"]
    X_scaled = scaler.transform(X)

    # -------------------------------------------------------------
    # 2. Model selection
    # -------------------------------------------------------------
    st.header("2️⃣ Select a Model")
    model_choice = st.selectbox("Choose a classification model:", list(models.keys()))
    model = models[model_choice]

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # -------------------------------------------------------------
    # 3. Evaluation metrics
    # -------------------------------------------------------------
    st.header("3️⃣ Evaluation Metrics")

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    # -------------------------------------------------------------
    # 4. Confusion matrix + classification report
    # -------------------------------------------------------------
    st.header("4️⃣ Confusion Matrix & Classification Report")

    col_left, col_right = st.columns(2)

    with col_left:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Malignant (0)", "Benign (1)"],
                    yticklabels=["Malignant (0)", "Benign (1)"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_choice}")
        st.pyplot(fig)

    with col_right:
        report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"])
        st.text("Classification Report")
        st.code(report)

    # -------------------------------------------------------------
    # 5. Compare all models at once (bonus)
    # -------------------------------------------------------------
    st.header("5️⃣ Compare All Models on This Data")
    if st.checkbox("Show comparison table for all 5 models"):
        rows = []
        for name, m in models.items():
            p = m.predict(X_scaled)
            pr = m.predict_proba(X_scaled)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_true, p), 4),
                "AUC": round(roc_auc_score(y_true, pr), 4),
                "Precision": round(precision_score(y_true, p), 4),
                "Recall": round(recall_score(y_true, p), 4),
                "F1": round(f1_score(y_true, p), 4),
                "MCC": round(matthews_corrcoef(y_true, p), 4),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

else:
    st.info("👆 Upload the `test_data.csv` file (included in this repo) to get started.")
