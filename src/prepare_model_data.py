
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/AirQuality_Cleaned.csv")

# Convert Timestamp back to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort observations chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Features available at the current hour
feature_cols = [
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]

# Target: benzene concentration one hour later
df["Target_Next_Hour"] = df["C6H6(GT)"].shift(-1)
df["Target_Timestamp"] = df["Timestamp"].shift(-1)

# Keep rows where the future target is available
model_df = df.dropna(subset=["Target_Next_Hour"]).copy()

# Retain features, target and timestamps
model_df = model_df[
    ["Timestamp"] + feature_cols
    + ["Target_Timestamp", "Target_Next_Hour"]
]

# Save without imputing missing feature values yet
model_df.to_csv(
    "data/AirQuality_Model_Input.csv",
    index=False
)

print("Model input shape:", model_df.shape)
print("\nMissing values:")
print(model_df.isna().sum())

print("\nFirst five records:")
print(model_df.head())

print("\nSaved: data/AirQuality_Model_Input.csv")
