# notebooks/03_gold.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, length, year, month,
    dayofweek, current_timestamp
)

spark = (SparkSession.builder
    .appName("gold_transformation")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

SILVER_PATH   = "data/silver/reviews"
GOLD_ML_PATH  = "data/gold/ml_features"
GOLD_NLP_PATH = "data/gold/nlp_text"

# ── 1. Read Silver ───────────────────────────────────────────────
df = spark.read.format("delta").load(SILVER_PATH)
print(f"Silver row count: {df.count()}")

# ════════════════════════════════════════════════════════════════
# GOLD BRANCH 1 — ML Features (XGBoost)
# Target: rating (1-5 multiclass)
# Features: structured/numeric/categorical only
# ════════════════════════════════════════════════════════════════
gold_ml = (df
    .withColumn("review_year",    year(col("review_date")))
    .withColumn("review_month",   month(col("review_date")))
    .withColumn("review_dow",     dayofweek(col("review_date")))
    .withColumn("do_recommend",   col("do_recommend").cast("int"))
    .withColumn("has_title",      when(col("review_title").isNull(), 0).otherwise(1))
    .select(
        # target
        col("rating").alias("label"),
        # numeric features
        "num_helpful",
        "review_text_len",
        "review_title_len",
        "do_recommend",
        "has_title",
        "review_year",
        "review_month",
        "review_dow",
        # categorical features (will be encoded in ML notebook)
        "brand",
        "primaryCategories",
        # audit
        "_silver_processed_at"
    )
    .withColumn("_gold_created_at", current_timestamp())
)

# drop rows where any feature is null
gold_ml = gold_ml.dropna(subset=[
    "label", "num_helpful", "review_text_len"
])

print(f"Gold ML row count  : {gold_ml.count()}")
gold_ml.printSchema()
gold_ml.show(3)

(gold_ml.write
    .format("delta")
    .mode("overwrite")
    .save(GOLD_ML_PATH))

print("✅ Gold ML table written")

# ════════════════════════════════════════════════════════════════
# GOLD BRANCH 2 — NLP Text (DistilBERT)
# Target: sentiment (0=negative, 1=positive)
# Input: review_text only
# Drop rating=3 (neutral — ambiguous for binary classification)
# ════════════════════════════════════════════════════════════════
gold_nlp = (df
    .filter(col("rating") != 3)
    .withColumn("sentiment",
        when(col("rating") >= 4, 1)
        .otherwise(0))
    .select(
        col("review_text").alias("text"),
        col("sentiment").alias("label"),
        # keep rating for reference only
        "rating",
        "_silver_processed_at"
    )
    .withColumn("_gold_created_at", current_timestamp())
    .dropna(subset=["text", "label"])
)

print(f"Gold NLP row count : {gold_nlp.count()}")
print("Sentiment distribution:")
gold_nlp.groupBy("label").count().orderBy("label").show()
gold_nlp.printSchema()
gold_nlp.show(3, truncate=80)

(gold_nlp.write
    .format("delta")
    .mode("overwrite")
    .save(GOLD_NLP_PATH))

print("✅ Gold NLP table written")
spark.stop()