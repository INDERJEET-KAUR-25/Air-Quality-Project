import pandas as pd

# Load the original dataset
df = pd.read_csv(
    "data/AirQualityUCI.csv",
    sep=";",
    decimal=","
)

# Remove completely empty rows and columns
df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")

# Keep the 15 actual dataset columns
df = df.iloc[:, :15].copy()

# Convert measurement columns to numeric values
measurement_cols = df.columns.drop(["Date", "Time"])

for col in measurement_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace the dataset's missing-value marker
df[measurement_cols] = df[measurement_cols].replace(-200, float("nan"))

# Combine date and time
df["Timestamp"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce"
)

# Remove rows with invalid timestamps
df = df.dropna(subset=["Timestamp"])

# Put Timestamp first
df = df.drop(columns=["Date", "Time"])
df = df[["Timestamp"] + [
    col for col in df.columns if col != "Timestamp"
]]

# Save a separate cleaned file
df.to_csv(
    "data/AirQuality_Cleaned.csv",
    index=False
)

print("Cleaned dataset shape:", df.shape)
print("Saved: data/AirQuality_Cleaned.csv")
print("\nMissing values by column:")
print(df.isna().sum())