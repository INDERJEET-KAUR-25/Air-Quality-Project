
from pathlib import Path

import pandas as pd  

# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "AirQuality_Model_Input.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model"

TARGET_COLUMN = "Target_Next_Hour"
TIMESTAMP_COLUMN = "Timestamp"
TRAIN_RATIO = 0.80

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("PREPARING SPARK ML PREDICTION DATASET")
    print("=" * 60)

    # --------------------------------------------------
    # 2. Load validated model input data
    # --------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run src/validate_timestamps.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput file: {INPUT_FILE.name}")
    print(f"Original dimensions: {df.shape}")
    print("\nAvailable columns:")
    print(df.columns.tolist())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' was not found. "
            "Check the column names printed above."
        )

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Expected timestamp column '{TIMESTAMP_COLUMN}' was not found. "
            "Check the column names printed above."
        )

    # --------------------------------------------------
    # 3. Sort observations chronologically
    # --------------------------------------------------

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN], errors="coerce"
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(
        subset=[TIMESTAMP_COLUMN, TARGET_COLUMN]
    ).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    print(f"\nRows after removing missing timestamps/targets: {len(df)}")
    print(f"Prediction target: {TARGET_COLUMN}")
    print("Meaning: benzene concentration one hour ahead")

    # --------------------------------------------------
    # 4. Identify numerical input features
    # --------------------------------------------------

    excluded_columns = {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        "Timestamp",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    # Convert all candidate features to numeric values.
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace invalid numeric values with missing values.
    df[feature_columns] = df[feature_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    if not feature_columns:
        raise ValueError("No input features were found.")

    print(f"\nCandidate input features: {len(feature_columns)}")
    print(feature_columns)

    # --------------------------------------------------
    # 5. Split chronologically: first 80%, last 20%
    # --------------------------------------------------

    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough rows for a train/test split.")

    print("\nChronological split:")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows:  {len(test_df)}")
    print(
        "Training period: "
        f"{train_df[TIMESTAMP_COLUMN].min()} to "
        f"{train_df[TIMESTAMP_COLUMN].max()}"
    )
    print(
        "Testing period:  "
        f"{test_df[TIMESTAMP_COLUMN].min()} to "
        f"{test_df[TIMESTAMP_COLUMN].max()}"
    )

    # --------------------------------------------------
    # 6. Remove features entirely missing in training
    # --------------------------------------------------

    usable_features = [
        column
        for column in feature_columns
        if train_df[column].notna().any()
    ]

    dropped_features = sorted(
        set(feature_columns) - set(usable_features)
    )

    if dropped_features:
        print("\nFeatures dropped because all training values are missing:")
        for column in dropped_features:
            print(f" - {column}")

    if not usable_features:
        raise ValueError(
            "No usable features remain after checking training data."
        )

    # --------------------------------------------------
    # 7. Impute using training-set medians only
    # --------------------------------------------------

    medians = train_df[usable_features].median()

    train_df[usable_features] = train_df[
        usable_features
    ].fillna(medians)

    test_df[usable_features] = test_df[
        usable_features
    ].fillna(medians)

    # A feature could have missing values in the test set
    # even if its training median was unavailable. Remove
    # any remaining rows with missing feature values.
    train_df = train_df.dropna(subset=usable_features)
    test_df = test_df.dropna(subset=usable_features)

    # --------------------------------------------------
    # 8. Keep timestamp as metadata, not as a model feature
    # --------------------------------------------------

    output_columns = [
        TIMESTAMP_COLUMN,
        *usable_features,
        TARGET_COLUMN,
    ]

    train_output = train_df[output_columns].copy()
    test_output = test_df[output_columns].copy()

    # --------------------------------------------------
    # 9. Save the datasets and feature list
    # --------------------------------------------------

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"
    feature_path = OUTPUT_DIR / "feature_columns.txt"

    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)

    feature_path.write_text(
        "\n".join(usable_features),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETED")
    print("=" * 60)

    print(f"\nFinal training rows: {len(train_output)}")
    print(f"Final testing rows:  {len(test_output)}")
    print(f"Final input features: {len(usable_features)}")
    print(f"Missing values in training data: {int(train_output.isna().sum().sum())}")
    print(f"Missing values in testing data:  {int(test_output.isna().sum().sum())}")

    print("\nFiles saved:")
    print(f" - {train_path.relative_to(PROJECT_ROOT)}")
    print(f" - {test_path.relative_to(PROJECT_ROOT)}")
    print(f" - {feature_path.relative_to(PROJECT_ROOT)}")

    print("\nFirst five training records:")
    print(train_output.head().to_string(index=False))

    print("\nTarget summary:")
    print(train_output[TARGET_COLUMN].describe())

    print("\nNext step: train the Spark MLlib Linear Regression model.")


if __name__ == "__main__":
    main()
