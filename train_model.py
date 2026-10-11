from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ----------------------------------
# 1. Set project paths
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sensor_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "irrigation_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# 2. Load dataset
# ----------------------------------
print("\nLoading SugarSmart dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that data/sensor_data.csv exists."
    )

df = pd.read_csv(DATA_PATH)

features = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall_mm",
    "crop_stage",
]

target = "irrigation_needed"

required_columns = features + [target]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df[required_columns].copy()

# ----------------------------------
# 3. Validate dataset
# ----------------------------------
df = df.dropna()

if df.empty:
    raise ValueError("Dataset is empty after removing missing values.")

df["irrigation_needed"] = pd.to_numeric(
    df["irrigation_needed"], errors="raise"
).astype(int)

if not set(df["irrigation_needed"].unique()).issubset({0, 1}):
    raise ValueError(
        "irrigation_needed must contain only 0 and 1."
    )

if df["irrigation_needed"].nunique() < 2:
    raise ValueError(
        "The dataset must contain both target classes: 0 and 1."
    )

X = df[features]
y = df[target]

print(f"Total usable records: {len(df)}")
print("\nTarget class distribution:")
print(y.value_counts().sort_index())

# ----------------------------------
# 4. Split dataset
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# ----------------------------------
# 5. Preprocess sensor data
# ----------------------------------
numeric_features = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall_mm",
]

categorical_features = ["crop_stage"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# ----------------------------------
# 6. Build Random Forest model
# ----------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    ]
)

# ----------------------------------
# 7. Train model
# ----------------------------------
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

# ----------------------------------
# 8. Evaluate on unseen test data
# ----------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

recall = recall_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

print("\n" + "=" * 45)
print("SUGARSMART MODEL EVALUATION")
print("=" * 45)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-score:          {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Irrigation", "Irrigation Needed"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------
# 9. Save trained model
# ----------------------------------
joblib.dump(model, MODEL_PATH)

print(f"\nTrained model saved to: {MODEL_PATH}")

# ----------------------------------
# 10. Save actual test metrics
# ----------------------------------
metrics = {
    "model": "RandomForestClassifier",
    "dataset": "sensor_data.csv",
    "total_records": int(len(df)),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "accuracy": round(float(accuracy), 4),
    "balanced_accuracy": round(float(balanced_accuracy), 4),
    "irrigation_precision": round(float(precision), 4),
    "irrigation_recall": round(float(recall), 4),
    "irrigation_f1_score": round(float(f1), 4),
    "confusion_matrix": confusion_matrix(
        y_test, y_pred
    ).tolist(),
    "random_state": 42,
    "test_size": 0.20,
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print(f"Metrics saved to: {METRICS_PATH}")
print("\nModel training completed successfully!")