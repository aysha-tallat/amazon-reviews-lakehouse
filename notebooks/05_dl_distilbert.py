# notebooks/05_dl_distilbert.py
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .appName("dl_distilbert")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

GOLD_NLP_PATH = "data/gold/nlp_text"
MLFLOW_URI    = "sqlite:///mlflow.db"
MODEL_NAME    = "distilbert-base-uncased"
MAX_LEN       = 128
BATCH_SIZE    = 16
EPOCHS        = 2
LR            = 2e-5
DEVICE        = torch.device("cpu")

# ── 1. Load Gold NLP table ───────────────────────────────────────
df = spark.read.format("delta").load(GOLD_NLP_PATH)
print(f"Gold NLP row count: {df.count()}")

pdf = df.select("text", "label").toPandas()
pdf = pdf.dropna(subset=["text", "label"])
pdf["label"] = pdf["label"].astype(int)
spark.stop()

print(f"Class distribution:\n{pdf['label'].value_counts()}")

# ── 2. Handle class imbalance ────────────────────────────────────
# Undersample majority class (positive) to 3x minority
neg_df = pdf[pdf["label"] == 0]
pos_df = pdf[pdf["label"] == 1].sample(
    n=min(len(neg_df) * 3, len(pdf[pdf["label"] == 1])),
    random_state=42
)
pdf = pd.concat([neg_df, pos_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced distribution:\n{pdf['label'].value_counts()}")

# ── 3. Train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    pdf["text"].tolist(),
    pdf["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=pdf["label"]
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. Tokenizer ─────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

train_dataset = ReviewDataset(X_train, y_train)
test_dataset  = ReviewDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── 5. Model ─────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS * len(train_loader)
)

# ── 6. Training loop ─────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("amazon-reviews-dl")

with mlflow.start_run(run_name="distilbert_sentiment"):

    mlflow.log_params({
        "model":      MODEL_NAME,
        "max_len":    MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs":     EPOCHS,
        "lr":         LR,
        "device":     str(DEVICE)
    })

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

    # ── 7. Evaluation ────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
          target_names=["negative", "positive"]))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.pytorch.log_model(model, "distilbert_model")

    print("✅ DistilBERT run logged to MLflow")