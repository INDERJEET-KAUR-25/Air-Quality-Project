
import pandas as pd

# Load cleaned data
df = pd.read_csv("data/AirQuality_Cleaned.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values("Timestamp").reset_index(drop=True)

# Check duplicate timestamps
duplicate_count = df["Timestamp"].duplicated().sum()

# Check time gaps between consecutive observations
time_diff = df["Timestamp"].diff()
gap_count = (time_diff > pd.Timedelta(hours=1)).sum()
non_hourly_count = (
    time_diff.notna()
    & (time_diff != pd.Timedelta(hours=1))
).sum()

print("Total cleaned rows:", len(df))
print("Duplicate timestamps:", duplicate_count)
print("Gaps longer than one hour:", gap_count)
print("Non-hourly intervals:", non_hourly_count)

print("\nMost common time intervals:")
print(time_diff.value_counts().head(10))

# Match each current reading to the reading exactly one hour later
target_df = df[["Timestamp", "C6H6(GT)"]].copy()

target_df["Timestamp"] = (
    target_df["Timestamp"] - pd.Timedelta(hours=1)
)

target_df = target_df.rename(
    columns={"C6H6(GT)": "Target_Next_Hour"}
)

# Merge the exact future target onto the current observation
model_df = df.merge(
    target_df,
    on="Timestamp",
    how="left",
    validate="one_to_one"
)

# Retain rows with a valid future target
model_df = model_df.dropna(
    subset=["Target_Next_Hour"]
).copy()

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

model_df = model_df[
    ["Timestamp"] + feature_cols + ["Target_Next_Hour"]
]

model_df.to_csv(
    "data/AirQuality_Model_Input.csv",
    index=False
)

print("\nExact one-hour model input shape:", model_df.shape)
print("Missing target values:", model_df["Target_Next_Hour"].isna().sum())
print("Saved: data/AirQuality_Model_Input.csv")
