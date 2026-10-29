
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
FEATURE_FILE = DATA_DIR / "feature_columns.txt"

MODEL_PATH = OUTPUT_DIR / "linear_regression_model"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions"
METRICS_PATH = OUTPUT_DIR / "model_metrics.txt"


def main():
    spark = (
        SparkSession.builder
        .appName("AirQualityBenzenePrediction")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("=" * 60)
        print("SPARK MLlib: BENZENE PREDICTION")
        print("=" * 60)

        # --------------------------------------------------
        # 2. Check required files
        # --------------------------------------------------

        for file_path in [TRAIN_FILE, TEST_FILE, FEATURE_FILE]:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {file_path}\n"
                    "Run src/prepare_spark_model_data.py first."
                )

        # --------------------------------------------------
        # 3. Read training and testing datasets
        # --------------------------------------------------

        train_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(TRAIN_FILE))
        )

        test_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(TEST_FILE))
        )

        feature_columns = [
            line.strip()
            for line in FEATURE_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        target_column = "Target_Next_Hour"
        timestamp_column = "Timestamp"

        print("\n1. Dataset information")
        print(f"Training rows: {train_df.count()}")
        print(f"Testing rows:  {test_df.count()}")
        print(f"Input features: {len(feature_columns)}")

        # --------------------------------------------------
        # 4. Rename feature columns to Spark-safe names
        # --------------------------------------------------
        # Some original names contain dots and parentheses.
        # Simple aliases avoid column-name interpretation issues.

        safe_features = []

        for index, original_name in enumerate(feature_columns, start=1):
            safe_name = f"feature_{index}"
            train_df = train_df.withColumnRenamed(
                original_name, safe_name
            )
            test_df = test_df.withColumnRenamed(
                original_name, safe_name
            )
            safe_features.append(safe_name)

        train_df = train_df.withColumnRenamed(
            target_column, "label"
        )
        test_df = test_df.withColumnRenamed(
            target_column, "label"
        )

        # Check that required fields exist.
        required_train = {"label", timestamp_column, *safe_features}
        required_test = {"label", timestamp_column, *safe_features}

        if not required_train.issubset(set(train_df.columns)):
            raise ValueError(
                "Training data is missing one or more required columns."
            )

        if not required_test.issubset(set(test_df.columns)):
            raise ValueError(
                "Testing data is missing one or more required columns."
            )

        # --------------------------------------------------
        # 5. Assemble features into a vector
        # --------------------------------------------------

        assembler = VectorAssembler(
            inputCols=safe_features,
            outputCol="features",
            handleInvalid="error"
        )

        train_vector = assembler.transform(train_df).select(
            "features", "label"
        )

        test_vector = assembler.transform(test_df).select(
            "features", "label"
        )

        print("\n2. Feature vectors created successfully.")

        # --------------------------------------------------
        # 6. Train Linear Regression model
        # --------------------------------------------------

        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            maxIter=100,
            regParam=0.0,
            elasticNetParam=0.0,
            solver="auto"
        )

        model = lr.fit(train_vector)

        print("\n3. Model training completed.")
        print(f"Iterations: {model.summary.totalIterations}")
        print(f"Training RMSE: {model.summary.rootMeanSquaredError:.4f}")
        print(f"Training R-squared: {model.summary.r2:.4f}")

        # --------------------------------------------------
        # 7. Predict on unseen test data
        # --------------------------------------------------

        predictions = model.transform(test_vector)

        print("\n4. Sample predictions")
        predictions.select(
            col("label").alias("Actual_Benzene"),
            col("prediction").alias("Predicted_Benzene")
        ).show(10, truncate=False)

        # --------------------------------------------------
        # 8. Evaluate test predictions
        # --------------------------------------------------

        metrics = {}

        for metric_name in ["rmse", "mae", "r2"]:
            evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName=metric_name
            )
            metrics[metric_name] = evaluator.evaluate(predictions)

        print("\n5. Test-set evaluation")
        print("-" * 40)
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"R-squared: {metrics['r2']:.4f}")

        # --------------------------------------------------
        # 9. Save predictions
        # --------------------------------------------------

        prediction_output = (
            predictions
            .select(
                col("label").alias("Actual_Benzene"),
                col("prediction").alias("Predicted_Benzene")
            )
            .withColumn(
                "Absolute_Error",
                col("Actual_Benzene") - col("Predicted_Benzene")
            )
        )

        prediction_output = prediction_output.withColumn(
            "Absolute_Error",
            col("Absolute_Error").cast("double")
        )

        # Make the error column absolute.
        from pyspark.sql.functions import abs as spark_abs

        prediction_output = prediction_output.withColumn(
            "Absolute_Error",
            spark_abs(col("Absolute_Error"))
        )

        (
            prediction_output.coalesce(1)
            .write.mode("overwrite")
            .option("header", True)
            .csv(str(PREDICTIONS_PATH))
        )

        # --------------------------------------------------
        # 10. Save trained model and evaluation metrics
        # --------------------------------------------------

        model.write().overwrite().save(str(MODEL_PATH))

        metrics_text = (
            "Air Quality Benzene Prediction - Test Metrics\n"
            "=============================================\n"
            f"Training rows: {train_df.count()}\n"
            f"Testing rows: {test_df.count()}\n"
            f"Input features: {len(safe_features)}\n"
            f"RMSE: {metrics['rmse']:.6f}\n"
            f"MAE: {metrics['mae']:.6f}\n"
            f"R-squared: {metrics['r2']:.6f}\n"
        )

        METRICS_PATH.write_text(
            metrics_text, encoding="utf-8"
        )

        # Save the mapping between original feature names and
        # their safe names for interpreting model coefficients.
        mapping_path = OUTPUT_DIR / "feature_mapping.txt"
        mapping_lines = [
            f"{safe}: {original}"
            for safe, original in zip(
                safe_features, feature_columns
            )
        ]
        mapping_path.write_text(
            "\n".join(mapping_lines),
            encoding="utf-8"
        )

        print("\n6. Saved outputs")
        print(f"Model: {MODEL_PATH}")
        print(f"Predictions: {PREDICTIONS_PATH}")
        print(f"Metrics: {METRICS_PATH}")
        print(f"Feature mapping: {mapping_path}")

        print("\nModel training and evaluation completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
