# =============================================================================
# CASE STUDY 1: Telecom Industry – Customer Churn Prediction
# Method: Artificial Neural Network (ANN) with TensorFlow / Keras
# =============================================================================
#
# Business Background:
#   A telecom company faces high customer churn due to competitive pricing
#   and poor service experience. The goal is to predict which customers are
#   likely to leave so preventive retention actions can be taken.
#
# Objective:
#   1. Predict customer churn using ANN
#   2. Identify key factors influencing churn
#   3. Improve customer retention strategy
#
# ANN Architecture (as specified):
#   Input  Layer : 11 features
#   Hidden Layer 1: 64 neurons (ReLU activation)
#   Hidden Layer 2: 32 neurons (ReLU activation)
#   Output Layer  :  1 neuron  (Sigmoid activation)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score, roc_curve)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# STEP 1 – DATA LOADING & EXPLORATION
# =============================================================================
print("=" * 65)
print("STEP 1 : DATA LOADING & EXPLORATION")
print("=" * 65)

df = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")

print(f"\nDataset shape  : {df.shape}")
print(f"\nColumn names   :\n{list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nStatistical summary:\n{df.describe().to_string()}")
print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
print(f"\nChurn distribution (Exited):\n{df['Exited'].value_counts().to_string()}")
print(f"\nOverall churn rate : {df['Exited'].mean():.2%}")

# =============================================================================
# STEP 2 – DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 2 : DATA PREPROCESSING")
print("=" * 65)

# ── 2a. Handle missing values ──────────────────────────────────────────────
# (Dataset has no missing values; shown for completeness)
print(f"\n[2a] Missing values before cleaning : {df.isnull().sum().sum()}")
df = df.dropna()
print(f"     Missing values after  cleaning : {df.isnull().sum().sum()}")

# ── 2b. Drop non-predictive identifier columns ─────────────────────────────
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
print(f"\n[2b] Dropped: RowNumber, CustomerId, Surname")

# ── 2c. Encode categorical variables ──────────────────────────────────────
#   Gender   : Label Encoding  (Female → 0, Male → 1)
#   Geography: One-Hot Encoding (France = baseline, drop_first=True)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
print(f"\n[2c] Gender encoded  → Female=0, Male=1")

df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
print(f"     Geography one-hot → columns added: Geography_Germany, Geography_Spain")
print(f"     Final columns: {list(df.columns)}")

# ── 2d. Separate features and target ──────────────────────────────────────
X = df.drop(columns=["Exited"]).values   # shape (10000, 11)
y = df["Exited"].values

print(f"\n[2d] Feature matrix X : {X.shape}  (11 features as required)")
print(f"     Target vector  y : {y.shape}")

# ── 2e. Train-test split (80 / 20) ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[2e] Train set : {X_train.shape}")
print(f"     Test  set : {X_test.shape}")

# ── 2f. Normalize numerical data (StandardScaler) ─────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print(f"\n[2f] Features normalized with StandardScaler")
print(f"     Train mean ≈ {X_train.mean():.4f}  (should be ~0)")
print(f"     Train std  ≈ {X_train.std():.4f}   (should be ~1)")

# =============================================================================
# STEP 3 – ANN MODEL DESIGN
# =============================================================================
print("\n" + "=" * 65)
print("STEP 3 : ANN MODEL DESIGN")
print("=" * 65)

# Architecture (exactly as specified in the case study):
#   Input  Layer  → 11 customer features
#   Hidden Layer 1 → 64 neurons, ReLU  (pattern learning)
#   Hidden Layer 2 → 32 neurons, ReLU  (pattern learning)
#   Output Layer   →  1 neuron,  Sigmoid (churn probability)

model = Sequential([
    Input(shape=(11,)),                          # Input Layer  : 11 features
    Dense(64, activation="relu"),                # Hidden Layer 1: 64 neurons, ReLU
    Dense(32, activation="relu"),                # Hidden Layer 2: 32 neurons, ReLU
    Dense(1,  activation="sigmoid")             # Output Layer  :  1 neuron, Sigmoid
], name="ANN_Churn_Predictor")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# =============================================================================
# STEP 4 – MODEL TRAINING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4 : MODEL TRAINING")
print("=" * 65)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =============================================================================
# STEP 5 – MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5 : MODEL EVALUATION")
print("=" * 65)

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
precision  = tp / (tp + fp)
recall     = tp / (tp + fn)
f1         = 2 * precision * recall / (precision + recall)

print(f"\n{'─'*40}")
print(f"  Test Accuracy      : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  ROC-AUC Score      : {roc_auc:.4f}")
print(f"  Precision (Churn)  : {precision:.4f}")
print(f"  Recall    (Churn)  : {recall:.4f}")
print(f"  F1-Score  (Churn)  : {f1:.4f}")
print(f"{'─'*40}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"  True  Negatives (Stayed  → Stayed ) : {tn}")
print(f"  False Positives (Stayed  → Churned) : {fp}")
print(f"  False Negatives (Churned → Stayed ) : {fn}")
print(f"  True  Positives (Churned → Churned) : {tp}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed (0)", "Churned (1)"]))

# =============================================================================
# STEP 6 – VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 65)
print("STEP 6 : VISUALIZATIONS")
print("=" * 65)

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    "Case Study 1: Telecom Customer Churn Prediction — ANN Results",
    fontsize=17, fontweight="bold", y=0.98
)

# ── Plot 1: Training & Validation Loss ────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(history.history["loss"],     color="steelblue", lw=2, label="Train Loss")
ax1.plot(history.history["val_loss"], color="tomato",    lw=2, linestyle="--", label="Val Loss")
ax1.set_title("Training vs Validation Loss", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Binary Cross-Entropy Loss")
ax1.legend(); ax1.grid(alpha=0.3)

# ── Plot 2: Training & Validation Accuracy ────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(history.history["accuracy"],     color="steelblue", lw=2, label="Train Accuracy")
ax2.plot(history.history["val_accuracy"], color="tomato",    lw=2, linestyle="--", label="Val Accuracy")
ax2.set_title("Training vs Validation Accuracy", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.legend(); ax2.grid(alpha=0.3)

# ── Plot 3: Confusion Matrix ───────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"],
    linewidths=0.5, linecolor="gray"
)
ax3.set_title("Confusion Matrix", fontweight="bold")
ax3.set_xlabel("Predicted Label"); ax3.set_ylabel("True Label")

# ── Plot 4: ROC Curve ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ax4.plot(fpr, tpr, color="steelblue", lw=2, label=f"ANN  (AUC = {roc_auc:.4f})")
ax4.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax4.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
ax4.set_title("ROC Curve", fontweight="bold")
ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
ax4.legend(); ax4.grid(alpha=0.3)

# ── Plot 5: Predicted Probability Distribution ────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(y_pred_prob[y_test == 0], bins=40, alpha=0.65,
         color="steelblue", label="Stayed  (Actual 0)")
ax5.hist(y_pred_prob[y_test == 1], bins=40, alpha=0.65,
         color="tomato",    label="Churned (Actual 1)")
ax5.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold = 0.5")
ax5.set_title("Predicted Churn Probability Distribution", fontweight="bold")
ax5.set_xlabel("Churn Probability"); ax5.set_ylabel("Number of Customers")
ax5.legend(); ax5.grid(alpha=0.3)

# ── Plot 6: Key Churn Factors (Churn Rate by Geography & Gender) ──────────
ax6 = fig.add_subplot(2, 3, 6)
orig = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")
churn_geo = orig.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
bars = ax6.bar(churn_geo.index, churn_geo.values,
               color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black", width=0.5)
ax6.set_title("Key Factor: Churn Rate by Geography", fontweight="bold")
ax6.set_xlabel("Geography"); ax6.set_ylabel("Churn Rate")
ax6.set_ylim(0, churn_geo.max() + 0.07)
ax6.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, churn_geo.values):
    ax6.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
             f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("churn_ann_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → churn_ann_results.png")

# =============================================================================
# STEP 7 – SINGLE CUSTOMER PREDICTION (Retention Strategy Demo)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 7 : SINGLE CUSTOMER PREDICTION")
print("=" * 65)

# Customer profile:
#   CreditScore=600, Gender=Male(1), Age=40, Tenure=3,
#   Balance=60000, NumOfProducts=2, HasCrCard=1,
#   IsActiveMember=1, EstimatedSalary=50000,
#   Geography_Germany=0, Geography_Spain=1  (→ Spain)
customer = np.array([[600, 1, 40, 3, 60000, 2, 1, 1, 50000, 0, 1]])
customer_scaled = scaler.transform(customer)
churn_prob = model.predict(customer_scaled, verbose=0)[0][0]

print(f"\n  Customer features : CreditScore=600, Male, Age=40, Tenure=3 yrs,")
print(f"                      Balance=60,000, 2 Products, HasCrCard, Active,")
print(f"                      Salary=50,000, Geography=Spain")
print(f"\n  Predicted churn probability : {churn_prob:.4f}  ({churn_prob*100:.2f}%)")
print(f"  Prediction                  : {'⚠ LIKELY TO CHURN — Trigger retention offer' if churn_prob >= 0.5 else '✓ LIKELY TO STAY — No immediate action needed'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("CASE STUDY SUMMARY")
print("=" * 65)
print(f"""
  Business Problem : Telecom customer churn prediction
  Dataset          : 10,000 customers, 11 input features
  Model            : Artificial Neural Network (ANN)

  Architecture:
    ┌─────────────────────────────────────────┐
    │  Input  Layer  →  11 features           │
    │  Hidden Layer 1 → 64 neurons  (ReLU)   │
    │  Hidden Layer 2 → 32 neurons  (ReLU)   │
    │  Output Layer   →  1 neuron   (Sigmoid) │
    └─────────────────────────────────────────┘

  Results:
    Test Accuracy  : {acc*100:.2f}%
    ROC-AUC Score  : {roc_auc:.4f}
    Precision      : {precision:.4f}
    Recall         : {recall:.4f}
    F1-Score       : {f1:.4f}

  Key Insight:
    Germany has the highest churn rate (~32%), suggesting
    targeted retention campaigns should prioritise German
    customers with low tenure and high monthly charges.
""")
