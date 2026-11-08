import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Bank Term-Deposit Predictor", page_icon="🏦", layout="wide")

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "y"


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def score_model(pipe, X, y_true):
    """Predict with a saved pipeline (no retraining) and return metrics + predictions."""
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }
    return metrics, pred


st.title("🏦 Term-Deposit Subscription Predictor")
st.caption("UCI Bank Marketing data · five classifiers loaded from saved pipelines · no retraining on upload")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help="Must contain all 16 input columns plus the target column 'y'.")
    chosen = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown("Positive class = **yes** (client subscribed).")

if uploaded is None:
    st.info("Upload `test_data.csv` from the sidebar to evaluate the models.")
    st.stop()

data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error(f"The uploaded file must contain the target column '{TARGET}'.")
    st.stop()

X = data.drop(columns=TARGET)
y = (data[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)

pipe = load_model(chosen)
expected = list(pipe.named_steps["prep"].feature_names_in_)
missing = [c for c in expected if c not in X.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()
X = X[expected]

st.subheader("Data preview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Input features", X.shape[1])
c3.metric("Subscribed (yes)", f"{y.mean():.1%}")
st.dataframe(data.head(10), use_container_width=True)

st.subheader(f"Results · {chosen}")
metrics, pred = score_model(pipe, X, y)

cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["no", "yes"], yticklabels=["no", "yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, pred, target_names=["no", "yes"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("All five models on this upload")
rows = []
for name in MODEL_FILES:
    m, _ = score_model(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comparison.style.highlight_max(axis=0, color="#c8e6c9"), use_container_width=True)
