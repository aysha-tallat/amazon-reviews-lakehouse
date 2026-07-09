# notebooks/06_consumption.py
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MLFLOW_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)

# ── 1. Fetch both runs from MLflow ───────────────────────────────
client = mlflow.tracking.MlflowClient()

def get_best_run(experiment_name):
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    return runs[0]

ml_run  = get_best_run("amazon-reviews-ml")
dl_run  = get_best_run("amazon-reviews-dl")

# ── 2. Build comparison table ────────────────────────────────────
comparison = pd.DataFrame({
    "Model": ["XGBoost (structured features)", "DistilBERT (review text)"],
    "Accuracy": [
        round(ml_run.data.metrics["accuracy"], 4),
        round(dl_run.data.metrics["accuracy"], 4)
    ],
    "F1 Weighted": [
        round(ml_run.data.metrics["f1_weighted"], 4),
        round(dl_run.data.metrics["f1_weighted"], 4)
    ],
    "Input Type": ["Tabular (numeric + categorical)", "Raw text (NLP)"],
    "Layer": ["Gold ML", "Gold NLP"]
})

print("\n" + "="*65)
print("       AMAZON REVIEWS LAKEHOUSE — MODEL COMPARISON")
print("="*65)
print(comparison.to_string(index=False))
print("="*65)

# ── 3. Visualisation ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Amazon Reviews Lakehouse — Consumption Layer",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

models    = comparison["Model"].tolist()
accuracy  = comparison["Accuracy"].tolist()
f1_scores = comparison["F1 Weighted"].tolist()
colors    = ["#4C72B0", "#DD8452"]

# Plot 1 — Accuracy comparison
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(["XGBoost", "DistilBERT"], accuracy, color=colors, width=0.5)
ax1.set_ylim(0, 1.1)
ax1.set_title("Accuracy Comparison")
ax1.set_ylabel("Accuracy")
for bar, val in zip(bars, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", fontweight="bold")

# Plot 2 — F1 comparison
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(["XGBoost", "DistilBERT"], f1_scores, color=colors, width=0.5)
ax2.set_ylim(0, 1.1)
ax2.set_title("F1 Score (Weighted) Comparison")
ax2.set_ylabel("F1 Score")
for bar, val in zip(bars2, f1_scores):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", fontweight="bold")

# Plot 3 — Pipeline architecture summary
ax3 = fig.add_subplot(gs[1, :])
ax3.axis("off")
pipeline = [
    "Raw CSVs\n(3 files, ~68K rows)",
    "Bronze\n(~74K rows)\nRaw Delta",
    "Silver\n(~32K rows)\nCleaned + Typed",
    "Gold ML\n(32K rows)\nStructured Features",
    "Gold NLP\n(31K rows)\nText + Sentiment",
    "XGBoost\nAcc: 0.74\nF1: 0.67",
    "DistilBERT\nAcc: 0.96\nF1: 0.96"
]
x_positions = [0.04, 0.17, 0.30, 0.50, 0.65, 0.50, 0.65]
y_positions = [0.50, 0.50, 0.50, 0.75, 0.75, 0.20, 0.20]
box_colors  = ["#f0f0f0","#d4e6f1","#a9cce3",
               "#a9dfbf","#a9dfbf","#4C72B0","#DD8452"]

for txt, x, y, c in zip(pipeline, x_positions, y_positions, box_colors):
    ax3.text(x, y, txt, transform=ax3.transAxes,
             fontsize=8, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=c,
                       edgecolor="gray", alpha=0.9))

ax3.set_title("Medallion Architecture Pipeline", fontweight="bold", pad=10)
ax3.text(0.5, -0.05,
         "Key Finding: Review text (DistilBERT) outperforms structured metadata "
         "(XGBoost) by +22% accuracy",
         transform=ax3.transAxes, ha="center", fontsize=10,
         fontstyle="italic", color="darkred")

plt.savefig("consumption_report.png", dpi=150, bbox_inches="tight")
print("\n✅ Consumption report saved: consumption_report.png")

# ── 4. Log comparison to MLflow ──────────────────────────────────
mlflow.set_experiment("amazon-reviews-consumption")
with mlflow.start_run(run_name="model_comparison"):
    mlflow.log_metric("xgb_accuracy",        accuracy[0])
    mlflow.log_metric("distilbert_accuracy", accuracy[1])
    mlflow.log_metric("xgb_f1",              f1_scores[0])
    mlflow.log_metric("distilbert_f1",       f1_scores[1])
    mlflow.log_artifact("consumption_report.png")
    print("✅ Comparison logged to MLflow")