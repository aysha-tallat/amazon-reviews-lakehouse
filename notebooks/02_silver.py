# notebooks/02_silver.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, lower, to_timestamp, length,
    when, regexp_replace, current_timestamp
)

spark = (SparkSession.builder
    .appName("silver_transformation")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

BRONZE_PATH = "data/bronze/reviews"
SILVER_PATH = "data/silver/reviews"

# ── 1. Read Bronze ──────────────────────────────────────────────
df = spark.read.format("delta").load(BRONZE_PATH)
print(f"Bronze row count: {df.count()}")

# ── 2. Select only useful columns ───────────────────────────────
df = df.select(
    "id",
    "name",
    "brand",
    "categories",
    "primaryCategories",
    "`reviews.date`",
    "`reviews.doRecommend`",
    "`reviews.numHelpful`",
    "`reviews.rating`",
    "`reviews.text`",
    "`reviews.title`",
    "`reviews.username`",
    "_ingested_at",
    "_source_file"
)
# ── 3. Rename columns (remove dots) ─────────────────────────────
df = (df
    .withColumnRenamed("reviews.date",        "review_date")
    .withColumnRenamed("reviews.doRecommend", "do_recommend")
    .withColumnRenamed("reviews.numHelpful",  "num_helpful")
    .withColumnRenamed("reviews.rating",      "rating")
    .withColumnRenamed("reviews.text",        "review_text")
    .withColumnRenamed("reviews.title",       "review_title")
    .withColumnRenamed("reviews.username",    "username"))

# ── 4. Drop nulls on critical columns ───────────────────────────
df = df.dropna(subset=["rating", "review_text"])

# ── 5. Type casting ──────────────────────────────────────────────
df = (df
    .withColumn("rating",      col("rating").cast("int"))
    .withColumn("num_helpful", col("num_helpful").cast("int"))
    .withColumn("review_date", to_timestamp(col("review_date")))
    .withColumn("do_recommend", col("do_recommend").cast("boolean")))

# ── 6. Filter valid ratings only (1-5) ──────────────────────────
df = df.filter(col("rating").between(1, 5))

# ── 7. Clean text fields ─────────────────────────────────────────
df = (df
    .withColumn("review_text",  trim(regexp_replace(col("review_text"),  r"[^\x00-\x7F]", " ")))
    .withColumn("review_title", trim(regexp_replace(col("review_title"), r"[^\x00-\x7F]", " ")))
    .withColumn("brand",        trim(lower(col("brand"))))
    .withColumn("name",         trim(col("name"))))

# ── 8. Engineer basic features ───────────────────────────────────
df = (df
    .withColumn("review_text_len",  length(col("review_text")))
    .withColumn("review_title_len", length(col("review_title")))
    .withColumn("num_helpful",      when(col("num_helpful").isNull(), 0)
                                    .otherwise(col("num_helpful"))))

# ── 9. Add silver audit column ───────────────────────────────────
df = df.withColumn("_silver_processed_at", current_timestamp())

# ── 10. Report ───────────────────────────────────────────────────
print(f"Silver row count : {df.count()}")
print(f"Silver col count : {len(df.columns)}")
df.printSchema()
df.show(3, truncate=80)

# ── 11. Write Silver Delta ───────────────────────────────────────
(df.write
    .format("delta")
    .mode("overwrite")
    .save(SILVER_PATH))

print("✅ Silver layer written")
spark.stop()