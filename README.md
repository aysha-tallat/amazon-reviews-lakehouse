# Amazon Reviews Lakehouse
End-to-end Medallion Architecture pipeline on Amazon product reviews — combining
PySpark, Delta Lake, XGBoost, and DistilBERT with MLflow experiment tracking.

---

## Architecture

```
Raw CSVs (3 files, ~68K rows)
        │
        ▼
┌─────────────┐
│   BRONZE    │  Raw ingestion — schema union, audit columns, Delta write
│  ~74K rows  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   SILVER    │  Clean, type-cast, dedupe, feature engineer → Delta
│  ~32K rows  │
└──────┬──────┘
       │
       ├─────────────────────┐
       ▼                     ▼
┌─────────────┐     ┌──────────────┐
│  GOLD · ML  │     │  GOLD · NLP  │
│ Structured  │     │  Text +      │
│ features    │     │  Sentiment   │
│ 32K rows    │     │  31K rows    │
└──────┬──────┘     └──────┬───────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌──────────────┐
│  XGBoost    │     │  DistilBERT  │
│  Rating     │     │  Sentiment   │
│  Prediction │     │  Fine-tune   │
└──────┬──────┘     └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 ▼
        ┌────────────────┐
        │  CONSUMPTION   │
        │  MLflow UI     │
        │  Flask API +   │
        │  Web UI        │
        └────────────────┘
```

---

## Results

| Model | Task | Input | Accuracy | F1 (Weighted) |
|---|---|---|---|---|
| XGBoost | Rating prediction (1–5) | Structured metadata | 0.7419 | 0.6738 |
| DistilBERT | Sentiment (pos/neg) | Raw review text | 0.9643 | 0.9643 |

**Key finding:** Review text alone (DistilBERT) outperforms all structured
metadata (XGBoost) by +22% accuracy — demonstrating the value of the NLP layer
in a lakehouse pipeline.

---

## Stack

| Layer | Tool | Version |
|---|---|---|
| Distributed compute | PySpark | 3.5.3 |
| Lakehouse storage | Delta Lake | 3.2.0 |
| ML model | XGBoost | ≥2.0.0 |
| DL model | DistilBERT (HuggingFace) | transformers ≥4.40.0 |
| Experiment tracking | MLflow | ≥2.13.0 |
| Serving | Flask | ≥3.0.0 |
| Language | Python | 3.11 |
| Java | OpenJDK | 17 |

---

## Project Structure

```
amazon-reviews-lakehouse/
├── data/
│   └── raw/                        # place downloaded CSVs here (gitignored)
├── notebooks/
│   ├── 01_bronze.py                # raw ingestion → Delta
│   ├── 02_silver.py                # cleaning + type casting → Delta
│   ├── 03_gold.py                  # feature split → ML + NLP Delta tables
│   ├── 04_ml_xgboost.py            # XGBoost training + MLflow logging
│   ├── 05_dl_distilbert.py         # DistilBERT fine-tuning + MLflow logging
│   ├── 06_consumption.py           # model comparison report
│   └── 07_inference.py             # Flask API + Web UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Consumer Reviews of Amazon Products** (Datafiniti)
- Source: [Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- 3 CSV files, ~68K rows, 21–24 columns per file
- Key fields: `reviews.rating`, `reviews.text`, `brand`, `primaryCategories`

Download and place all 3 CSVs in `data/raw/` before running the pipeline.

---

## Quickstart

### 1. Environment setup

```bash
conda create -n bigdata python=3.11 -y
conda activate bigdata

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export JAVA_TOOL_OPTIONS="--add-opens=java.base/javax.security.auth=ALL-UNNAMED"

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Run pipeline (in order)

```bash
python notebooks/01_bronze.py
python notebooks/02_silver.py
python notebooks/03_gold.py
python notebooks/04_ml_xgboost.py
python notebooks/05_dl_distilbert.py
python notebooks/06_consumption.py
```

### 3. Launch inference UI

```bash
python notebooks/07_inference.py
# open http://127.0.0.1:5000
```

### 4. View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# open http://127.0.0.1:5000
```

---

## MLflow Experiments

| Experiment | Run | Key Metric |
|---|---|---|
| amazon-reviews-ml | xgboost_rating_prediction | accuracy=0.7419 |
| amazon-reviews-dl | distilbert_sentiment | accuracy=0.9643 |
| amazon-reviews-consumption | model_comparison | both models compared |

---

## Notes

- DistilBERT trained on CPU — expect ~30 mins for 2 epochs on ~6K balanced samples
- Class imbalance in NLP branch handled via undersampling (positive:negative = 3:1)
- Delta tables are gitignored — rerun pipeline from `01_bronze.py` to regenerate
- MLflow DB (`mlflow.db`) is gitignored — experiments regenerate on rerun