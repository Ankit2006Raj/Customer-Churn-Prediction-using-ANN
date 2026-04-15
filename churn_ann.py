# =============================================================================
# Customer Churn Prediction using Artificial Neural Network (ANN)
# Dataset: Bank Customer Churn (10,000 records)
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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# 1. LOAD & EXPLORE DATA
# =============================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nChurn distribution:\n{df['Exited'].value_counts()}")
print(f"Churn rate: {df['Exited'].mean():.2%}")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# Drop irrelevant columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Encode categorical features
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])          # Female=0, Male=1

# One-hot encode Geography
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)  # France=baseline

print(f"Columns after encoding: {list(df.columns)}")

# Features & target
X = df.drop(columns=["Exited"]).values
y = df["Exited"].values

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Features: {X.shape[1]}")

# =============================================================================
# 3. BUILD ANN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("3. BUILDING ANN MODEL")
print("=" * 60)

model = Sequential([
    # Input + Hidden Layer 1
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden Layer 2
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    # Hidden Layer 3
    Dense(16, activation="relu"),
    Dropout(0.1),

    # Output Layer
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================================================================
# 4. TRAIN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("4. TRAINING MODEL")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# 5. EVALUATE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("5. EVALUATION")
print("=" * 60)

y_pred_prob = model.predict(X_test).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Test Accuracy : {acc:.4f}")
print(f"ROC-AUC Score : {roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Stayed','Churned'])}")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("ANN Customer Churn Prediction — Results", fontsize=16, fontweight="bold")

# --- (a) Training & Validation Loss ---
ax = axes[0, 0]
ax.plot(history.history["loss"],     label="Train Loss",      color="steelblue")
ax.plot(history.history["val_loss"], label="Validation Loss", color="tomato", linestyle="--")
ax.set_title("Training vs Validation Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# --- (b) Training & Validation Accuracy ---
ax = axes[0, 1]
ax.plot(history.history["accuracy"],     label="Train Accuracy",      color="steelblue")
ax.plot(history.history["val_accuracy"], label="Validation Accuracy", color="tomato", linestyle="--")
ax.set_title("Training vs Validation Accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(alpha=0.3)

# --- (c) Confusion Matrix ---
ax = axes[0, 2]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"])
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

# --- (d) ROC Curve ---
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(); ax.grid(alpha=0.3)

# --- (e) Prediction Probability Distribution ---
ax = axes[1, 1]
ax.hist(y_pred_prob[y_test == 0], bins=40, alpha=0.6, color="steelblue", label="Stayed")
ax.hist(y_pred_prob[y_test == 1], bins=40, alpha=0.6, color="tomato",    label="Churned")
ax.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")
ax.set_title("Predicted Probability Distribution")
ax.set_xlabel("Churn Probability"); ax.set_ylabel("Count")
ax.legend(); ax.grid(alpha=0.3)

# --- (f) Churn Rate by Geography (original data) ---
ax = axes[1, 2]
orig = pd.read_csv("data/Artificial_Neural_Network_Case_Study_data.csv")
churn_geo = orig.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
churn_geo.plot(kind="bar", ax=ax, color=["tomato", "steelblue", "seagreen"], edgecolor="black")
ax.set_title("Churn Rate by Geography")
ax.set_xlabel("Geography"); ax.set_ylabel("Churn Rate")
ax.set_xticklabels(churn_geo.index, rotation=0)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(churn_geo):
    ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("churn_ann_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → churn_ann_results.png")

# =============================================================================
# 7. SINGLE PREDICTION EXAMPLE
# =============================================================================
print("\n" + "=" * 60)
print("6. SINGLE PREDICTION EXAMPLE")
print("=" * 60)

# Example: CreditScore=600, Gender=Male(1), Age=40, Tenure=3,
#          Balance=60000, NumOfProducts=2, HasCrCard=1,
#          IsActiveMember=1, EstimatedSalary=50000,
#          Geography_Germany=0, Geography_Spain=1
sample = np.array([[600, 1, 40, 3, 60000, 2, 1, 1, 50000, 0, 1]])
sample_scaled = scaler.transform(sample)
prob = model.predict(sample_scaled)[0][0]
print(f"Churn probability: {prob:.4f}")
print(f"Prediction: {'CHURNED' if prob >= 0.5 else 'STAYED'}")

print("\nDone.")
