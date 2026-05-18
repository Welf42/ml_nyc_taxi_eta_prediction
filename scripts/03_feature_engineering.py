#!/usr/bin/env python3
"""
Feature engineering on the cleaned training data.

Features added:

  Spatial
  -------
  haversine_km       — great-circle distance between pickup and dropoff (km).
                       A proxy for trip length; correlated with duration but
                       does not leak the target.
  bearing_deg        — compass bearing from pickup to dropoff (0–360°).
                       Captures directionality (e.g. airport runs go one way).

  Temporal
  --------
  pickup_hour        — 0–23; captures within-day demand and congestion patterns.
  pickup_weekday     — 0 (Mon) – 6 (Sun).
  pickup_month       — 1–6 (dataset covers Jan–Jun 2016).
  is_weekend         — True for Saturday/Sunday.
  is_rush_hour       — True for 7–9 h and 17–19 h on weekdays.

  Categorical encoding
  --------------------
  store_and_fwd_flag — 'Y'/'N' → 1/0.

  Target transformation
  ---------------------
  log_trip_duration  — log1p(trip_duration). Right-skewed targets hurt
                       squared-error metrics; the log makes residuals more
                       symmetric and reduces the influence of extreme values.

Input : data/processed/train_clean.csv
Output: data/processed/train_features.csv

Run from the project root:
    python3 scripts/03_feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
OUT_PATH = ROOT / "data" / "processed" / "train_features.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(IN_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])
print(f"Loaded {len(df):,} rows")

# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometres."""
    R = 6_371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2 (0–360°)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


df["haversine_km"] = haversine_km(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"],
)
df["bearing_deg"] = bearing_deg(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"],
)

# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

df["pickup_hour"]    = df["pickup_datetime"].dt.hour
df["pickup_weekday"] = df["pickup_datetime"].dt.weekday   # 0 = Monday
df["pickup_month"]   = df["pickup_datetime"].dt.month

df["is_weekend"] = df["pickup_weekday"] >= 5

# Rush hour: 07:00–08:59 and 17:00–18:59, weekdays only
rush = df["pickup_hour"].isin(range(7, 9)) | df["pickup_hour"].isin(range(17, 19))
df["is_rush_hour"] = rush & ~df["is_weekend"]

# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)

# ---------------------------------------------------------------------------
# Target transformation
# ---------------------------------------------------------------------------

df["log_trip_duration"] = np.log1p(df["trip_duration"])

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

print("\n--- Feature snapshot (first 3 rows) ---")
feature_cols = [
    "haversine_km", "bearing_deg",
    "pickup_hour", "pickup_weekday", "pickup_month",
    "is_weekend", "is_rush_hour",
    "store_and_fwd_flag",
    "log_trip_duration",
]
print(df[feature_cols].head(3).to_string())

print("\n--- haversine_km summary ---")
print(df["haversine_km"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_string())

zero_dist = (df["haversine_km"] == 0).sum()
print(f"\nTrips with zero distance: {zero_dist:,} ({zero_dist / len(df):.4%})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df):,} rows with {len(df.columns)} columns to {OUT_PATH.name}")
print(f"Columns: {list(df.columns)}")
