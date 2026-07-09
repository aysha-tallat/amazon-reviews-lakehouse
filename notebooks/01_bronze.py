# notebooks/01_bronze.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name

spark = (SparkSession.builder
    .appName("bronze_ingestion")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

RAW_PATH   = "data/raw"
BRONZE_PATH = "data/bronze/reviews"

df = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("mergeSchema", "true")
    .csv(RAW_PATH))

df = (df
    .withColumn("_ingested_at", current_timestamp())
    .withColumn("_source_file", input_file_name()))

print(f"Bronze row count : {df.count()}")
print(f"Bronze col count : {len(df.columns)}")
df.printSchema()

(df.write
    .format("delta")
    .mode("overwrite")
    .save(BRONZE_PATH))

print("✅ Bronze layer written")
spark.stop()