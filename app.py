import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, roc_curve,
                             roc_auc_score, accuracy_score)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🏦", layout="wide")

st.title("🏦 Customer Churn Prediction using ANN")
st.markdown("Predict whether a bank customer will churn using an Artificial Neural Network.")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    X = df.drop(columns=["Exited"]).values
    y = df["Exited"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ANN using scikit-learn (same architecture: 64 → 32 → 1)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test

with st.spinner("Training ANN model... please wait ⏳"):
    model, scaler, X_test, y_test = load_and_train()

y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred      = (y_pred_prob >= 0.5).astype(int)
acc         = accuracy_score(y_test, y_pred)
roc_auc     = roc_auc_score(y_test, y_pred_prob)
cm          = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
f1 = 2*tp / (2*tp + fp + fn)

# ── Metrics ──────────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("Test Accuracy",    f"{acc*100:.2f}%")
c2.metric("ROC-AUC Score",    f"{roc_auc:.4f}")
c3.metric("F1-Score (Churn)", f"{f1:.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
st.subheader("📈 Visualizations")
col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"])
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

with col3:
    orig = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")
    fig, ax = plt.subplots()
    churn_geo = orig.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
    ax.bar(churn_geo.index, churn_geo.values,
           color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")
    for i, v in enumerate(churn_geo.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_title("Churn Rate by Geography")
    ax.set_ylabel("Churn Rate"); ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

# ── Single Prediction ─────────────────────────────────────────────────────────
st.subheader("🔍 Predict for a Single Customer")

col1, col2, col3 = st.columns(3)
with col1:
    credit_score = st.slider("Credit Score", 300, 850, 600)
    age          = st.slider("Age", 18, 92, 40)
    tenure       = st.slider("Tenure (years)", 0, 10, 3)
    balance      = st.number_input("Balance", 0.0, 250000.0, 60000.0)
with col2:
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card  = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    is_active    = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x else "No")
    salary       = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
with col3:
    gender    = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

gender_enc  = 1 if gender == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain   = 1 if geography == "Spain" else 0

if st.button("🚀 Predict Churn"):
    sample = np.array([[credit_score, gender_enc, age, tenure,
                        balance, num_products, has_cr_card,
                        is_active, salary, geo_germany, geo_spain]])
    sample_scaled = scaler.transform(sample)
    prob = model.predict_proba(sample_scaled)[0][1]

    st.markdown("---")
    if prob >= 0.5:
        st.error(f"⚠️ **LIKELY TO CHURN** — Churn Probability: **{prob*100:.2f}%**")
    else:
        st.success(f"✅ **LIKELY TO STAY** — Churn Probability: **{prob*100:.2f}%**")
    st.progress(float(prob))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Made by **Ankit Raj** &nbsp;|&nbsp; "
    "[GitHub](https://github.com/Ankit2006Raj) &nbsp;|&nbsp; "
    "[LinkedIn](https://www.linkedin.com/in/ankit-raj-226a36309) &nbsp;|&nbsp; "
    "📧 ankit9905163014@gmail.com"
)
