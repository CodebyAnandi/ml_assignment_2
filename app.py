"""
Streamlit App — Credit Card Default Prediction
------------------------------------------------
Upload the provided test_data.csv (or any CSV with the same 23 feature
columns + a 'target' column) and compare 5 trained classification models
that predict whether a credit card client will default next month.

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
    confusion_matrix, classification_report, RocCurveDisplay
)

st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide",
)

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

# =======================================================================
# SIDEBAR — controls
# =======================================================================
with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader(
        "Upload evaluation CSV (test data)",
        type=["csv"],
        help="Must contain the 23 feature columns + a 'target' column. "
             "Use the bundled test_data.csv if you don't have your own.",
    )

    st.markdown("---")
    model_choice = st.selectbox("Evaluate with this model", list(models.keys()))
    st.caption(f"Active model: **{model_choice}**")

    st.markdown("---")
    st.markdown(
        """
        **About this dataset**
        UCI *Default of Credit Card Clients* — 30,000 Taiwanese credit
        card holders, 23 features (credit limit, repayment history,
        bill amounts, demographics), predicting default next month.

        Held-out test set: 6,000 rows (20% split, `random_state=42`),
        never seen during training.
        """
    )

# =======================================================================
# MAIN AREA — header
# =======================================================================
st.title("💳 Credit Card Default Prediction")
st.markdown(
    "Predict whether a credit card client will default on payment next "
    "month (`target = 1`) using five classical classifiers trained on the "
    "**UCI Default of Credit Card Clients** dataset."
)

if uploaded_file is None:
    st.info("👈 Upload `test_data.csv` from the sidebar to see live results, "
            "or download it from the project repository.")
    st.stop()

# =======================================================================
# Load + validate uploaded data
# =======================================================================
df = pd.read_csv(uploaded_file)

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

model = models[model_choice]
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

st.success(f"Currently evaluating with **{model_choice}** on {df.shape[0]} rows "
           f"from your uploaded file.")

# =======================================================================
# TABS — Metrics / Confusion Matrix / Compare All Models
# =======================================================================
tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🧩 Confusion Matrix & Report", "⚖️ Compare All Models"])

# -----------------------------------------------------------------------
# Tab 1: Metrics
# -----------------------------------------------------------------------
with tab1:
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

    st.markdown("---")
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax_roc)
    ax_roc.set_title(f"ROC Curve — {model_choice}")
    st.pyplot(fig_roc)

    # Feature importance (only for tree-based models)
    if hasattr(model, "feature_importances_"):
        st.subheader("Top 10 Feature Importances")
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top10 = importances.sort_values(ascending=False).head(10)
        st.bar_chart(top10)

# -----------------------------------------------------------------------
# Tab 2: Confusion matrix + classification report
# -----------------------------------------------------------------------
with tab2:
    col_left, col_right = st.columns(2)

    with col_left:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["No Default (0)", "Default (1)"],
                    yticklabels=["No Default (0)", "Default (1)"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_choice}")
        st.pyplot(fig)

    with col_right:
        report = classification_report(
            y_true, y_pred, target_names=["No Default", "Default"]
        )
        st.text("Classification Report")
        st.code(report)

# -----------------------------------------------------------------------
# Tab 3: Compare all models
# -----------------------------------------------------------------------
with tab3:
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
    comp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comp_df, use_container_width=True)
    st.bar_chart(comp_df[["F1", "MCC"]])

    st.caption(
        "⚠️ Note: this dataset is imbalanced (~22% default rate). "
        "Accuracy alone can be misleading here — F1 and MCC give a "
        "fairer picture of how well a model catches actual defaulters."
    )
