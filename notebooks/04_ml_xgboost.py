# notebooks/04_ml_xgboost.py
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

spark = (SparkSession.builder
    .appName("ml_xgboost")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

GOLD_ML_PATH = "data/gold/ml_features"
MLFLOW_URI = "sqlite:///mlflow.db"

# ── 1. Load Gold ML table ────────────────────────────────────────
df = spark.read.format("delta").load(GOLD_ML_PATH)
print(f"Gold ML row count: {df.count()}")

# Convert to pandas for sklearn/xgboost
pdf = df.drop("_silver_processed_at", "_gold_created_at").toPandas()

# ── 2. Encode categoricals ───────────────────────────────────────
cat_cols = ["brand", "primaryCategories"]
encoders = {}

for c in cat_cols:
    le = LabelEncoder()
    pdf[c] = pdf[c].fillna("unknown")
    pdf[c] = le.fit_transform(pdf[c].astype(str))
    encoders[c] = le

# fill remaining nulls
pdf = pdf.fillna(0)

# ── 3. Split ─────────────────────────────────────────────────────
X = pdf.drop(columns=["label"])
y = pdf["label"] - 1          # XGBoost needs 0-indexed classes (0-4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── 4. Train XGBoost ─────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("amazon-reviews-ml")

params = {
    "objective":        "multi:softmax",
    "num_class":        5,
    "max_depth":        6,
    "learning_rate":    0.1,
    "n_estimators":     200,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "mlogloss",
    "random_state":     42
}

with mlflow.start_run(run_name="xgboost_rating_prediction"):

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── 5. Evaluate ──────────────────────────────────────────────
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["1★","2★","3★","4★","5★"]))

    # ── 6. Log to MLflow ─────────────────────────────────────────
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.xgboost.log_model(model, "xgboost_model")

    # ── 7. Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["1★","2★","3★","4★","5★"],
                yticklabels=["1★","2★","3★","4★","5★"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("xgb_confusion_matrix.png")
    mlflow.log_artifact("xgb_confusion_matrix.png")
    print("✅ Confusion matrix saved")

    # ── 8. Feature importance ────────────────────────────────────
    fi = pd.Series(model.feature_importances_, index=X.columns)
    fi = fi.sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fi.plot(kind="bar", ax=ax2, color="steelblue")
    ax2.set_title("XGBoost — Feature Importance")
    ax2.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig("xgb_feature_importance.png")
    mlflow.log_artifact("xgb_feature_importance.png")
    print("✅ Feature importance saved")

print(f"\n✅ XGBoost run logged to MLflow")
spark.stop()