
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, min, max, to_timestamp, date_format
)

# 1. Create Spark session
spark = (
    SparkSession.builder
    .appName("AirQualityDistributedAnalytics")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("DISTRIBUTED AIR QUALITY ANALYTICS")
print("=" * 60)

# 2. Load cleaned dataset
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/AirQuality_Cleaned.csv")
)

# 3. Inspect dataset
print("\n1. Dataset schema:")
df.printSchema()

print("\n2. Dataset dimensions:")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\n3. First five records:")
df.show(5, truncate=False)

# 4. Convert measurement columns to numeric safely.
# Backticks protect special characters such as dots in column names.
measurement_cols = [
    c for c in df.columns if c != "Timestamp"
]

for c in measurement_cols:
    df = df.withColumn(
        c,
        col(f"`{c}`").cast("double")
    )

# Replace the original missing-value marker, if present
for c in measurement_cols:
    df = df.withColumn(
        c,
        __import__("pyspark.sql.functions", fromlist=["when"])
        .when(col(f"`{c}`") == -200, None)
        .otherwise(col(f"`{c}`"))
    )

# 5. Register Spark SQL view
df.createOrReplaceTempView("air_quality")

# 6. Overall pollutant statistics
print("\n4. Overall pollutant statistics:")

df.select(
    avg(col("`CO(GT)`")).alias("Average_CO"),
    min(col("`CO(GT)`")).alias("Minimum_CO"),
    max(col("`CO(GT)`")).alias("Maximum_CO"),
    avg(col("`NO2(GT)`")).alias("Average_NO2"),
    avg(col("`C6H6(GT)`")).alias("Average_Benzene")
).show(truncate=False)

# 7. Spark SQL hourly aggregation
print("\n5. Hourly pollution readings:")

hourly_results = spark.sql("""
    SELECT
        date_format(Timestamp, 'yyyy-MM-dd HH:00:00') AS observation_hour,
        AVG(`C6H6(GT)`) AS average_benzene,
        AVG(`CO(GT)`) AS average_co
    FROM air_quality
    GROUP BY date_format(Timestamp, 'yyyy-MM-dd HH:00:00')
    ORDER BY observation_hour
""")

hourly_results.show(10, truncate=False)

# 8. Save output
hourly_results.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("outputs/hourly_analysis")

print("\nHourly analysis saved to outputs/hourly_analysis")
print("Spark analysis completed successfully.")

spark.stop()
