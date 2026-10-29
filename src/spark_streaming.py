
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, from_json,sqrt, pow, abs as spark_abs
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
)
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler


# --------------------------------------------------
# 1. Project paths and Kafka configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = str(
    PROJECT_ROOT / "outputs" / "linear_regression_model"
)

FEATURE_MAPPING_PATH = (
    PROJECT_ROOT / "outputs" / "feature_mapping.txt"
)

CHECKPOINT_DIR = str(
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "air_quality_predictions"
)

KAFKA_SERVERS = "localhost:9092"
KAFKA_TOPIC = "air-quality"


# --------------------------------------------------
# 2. Kafka JSON message schema
# --------------------------------------------------

schema = StructType([
    StructField("Timestamp", StringType(), True),
    StructField("CO(GT)", DoubleType(), True),
    StructField("PT08.S1(CO)", DoubleType(), True),
    StructField("C6H6(GT)", DoubleType(), True),
    StructField("PT08.S2(NMHC)", DoubleType(), True),
    StructField("NOx(GT)", DoubleType(), True),
    StructField("PT08.S3(NOx)", DoubleType(), True),
    StructField("NO2(GT)", DoubleType(), True),
    StructField("PT08.S4(NO2)", DoubleType(), True),
    StructField("PT08.S5(O3)", DoubleType(), True),
    StructField("T", DoubleType(), True),
    StructField("RH", DoubleType(), True),
    StructField("AH", DoubleType(), True),
    StructField("Target_Next_Hour", DoubleType(), True),
])


# --------------------------------------------------
# 3. Safe column reference
# --------------------------------------------------

def safe_col(name):
    """Handle column names containing dots and special characters."""
    escaped_name = name.replace("`", "``")
    return col(f"`{escaped_name}`")


# --------------------------------------------------
# 4. Load feature names in training order
# --------------------------------------------------

def load_feature_columns():
    if not FEATURE_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Feature mapping not found: {FEATURE_MAPPING_PATH}"
        )

    lines = FEATURE_MAPPING_PATH.read_text(
        encoding="utf-8"
    ).splitlines()

    features = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if ":" in line:
            prefix, remainder = line.split(":", 1)
            prefix = prefix.strip()

            if (
                prefix.lower().startswith("feature_")
                or prefix.isdigit()
            ):
                line = remainder.strip()

        if line and line not in features:
            features.append(line)

    if not features:
        raise ValueError(
            "No feature names found in feature_mapping.txt"
        )

    return features


# --------------------------------------------------
# 5. Process each streaming micro-batch
# --------------------------------------------------


def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        print(f"Batch {batch_id}: no valid records.")
        return

    print("\n" + "=" * 70)
    print(f"AIR QUALITY PREDICTION BATCH: {batch_id}")
    print("=" * 70)

    # Display individual predictions
    print("\nIndividual predictions:")

    batch_df.select(
        safe_col("Timestamp"),
        safe_col("Actual_Next_Hour"),
        safe_col("Predicted_Next_Hour"),
    ).show(20, truncate=False)

    # Calculate prediction errors
    error_df = (
        batch_df
        .withColumn(
            "Absolute_Error",
            spark_abs(
                col("Actual_Next_Hour")
                - col("Predicted_Next_Hour")
            ),
        )
        .withColumn(
            "Squared_Error",
            pow(
                col("Actual_Next_Hour")
                - col("Predicted_Next_Hour"),
                2,
            ),
        )
    )

    # Calculate MAE and RMSE for this micro-batch
    metrics = error_df.agg(
        count("*").alias("Prediction_Count"),
        avg("Actual_Next_Hour").alias("Average_Actual"),
        avg("Predicted_Next_Hour").alias("Average_Predicted"),
        avg("Absolute_Error").alias("MAE"),
        sqrt(avg("Squared_Error")).alias("RMSE"),
    )

    print("\nBatch evaluation metrics:")
    metrics.show(truncate=False)



# --------------------------------------------------
# 6. Main streaming application
# --------------------------------------------------

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    spark = (
        SparkSession.builder
        .appName("AirQualityKafkaMLPredictions")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9",
        )
        .getOrCreate()
    )

    query = None

    try:
        spark.sparkContext.setLogLevel("WARN")

        # Load the model and the feature order.
        feature_columns = load_feature_columns()

        print("Loading trained Spark MLlib model...")

        model = LinearRegressionModel.load(MODEL_PATH)

        print("Model loaded successfully.")
        print(f"Model features ({len(feature_columns)}):")
        print(feature_columns)

        # Validate the incoming schema.
        available_columns = set(schema.fieldNames())

        missing_features = [
            feature
            for feature in feature_columns
            if feature not in available_columns
        ]

        if missing_features:
            raise ValueError(
                "Features missing from Kafka schema: "
                + ", ".join(missing_features)
            )

        # Read messages from Kafka.
        kafka_df = (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_SERVERS)
            .option("subscribe", KAFKA_TOPIC)
            .option("startingOffsets", "earliest")
            .option("maxOffsetsPerTrigger", 10)
            .load()
        )

        # Parse JSON messages.
        parsed_df = (
            kafka_df
            .select(
                from_json(
                    col("value").cast("string"),
                    schema,
                ).alias("data")
            )
            .select("data.*")
            .filter(safe_col("Timestamp").isNotNull())
        )

        # Remove rows with missing model input values.
        valid_condition = None

        for feature in feature_columns:
            condition = safe_col(feature).isNotNull()

            if valid_condition is None:
                valid_condition = condition
            else:
                valid_condition = valid_condition & condition

        valid_df = parsed_df.filter(valid_condition)

        # Rename special-character columns to simple names.
        # This prevents Spark from interpreting dots as nested fields.
        rename_map = {
            original_name: f"feature_{i + 1}"
            for i, original_name in enumerate(feature_columns)
        }

        assembler_df = valid_df.select(
            *[
                safe_col(original_name).alias(simple_name)
                for original_name, simple_name in rename_map.items()
            ],
            safe_col("Timestamp"),
            safe_col("Target_Next_Hour"),
        )

        # Assemble features in the same order as model training.
        assembler = VectorAssembler(
            inputCols=list(rename_map.values()),
            outputCol="features",
            handleInvalid="skip",
        )

        feature_df = assembler.transform(assembler_df)

        # Apply the trained regression model.
        prediction_df = model.transform(feature_df)

        # Prepare output for each micro-batch.
        output_df = prediction_df.select(
            safe_col("Timestamp"),
            safe_col("Target_Next_Hour").alias(
                "Actual_Next_Hour"
            ),
            safe_col("prediction").alias(
                "Predicted_Next_Hour"
            ),
        )

        # Start streaming predictions.
        query = (
            output_df.writeStream
            .foreachBatch(process_batch)
            .outputMode("append")
            .option("checkpointLocation", CHECKPOINT_DIR)
            .trigger(processingTime="5 seconds")
            .start()
        )

        print("\nStreaming prediction application started.")
        print(f"Kafka topic: {KAFKA_TOPIC}")
        print("Press Ctrl+C to stop.")

        query.awaitTermination()

    except KeyboardInterrupt:
        print("\nStopping the streaming application...")

    finally:
        if query is not None and query.isActive:
            query.stop()

        spark.stop()
        print("Spark streaming application stopped.")


if __name__ == "__main__":
    main()
