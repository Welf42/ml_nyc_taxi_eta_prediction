#!/usr/bin/env python3
"""
Data exploration for the NYC Taxi Trip Duration dataset.

Questions answered here:
- What does one row represent, and what columns exist in train vs. test?
- How is `trip_duration` distributed? Is it skewed?
- Are there temporal patterns in pickup volume?
- Are there anomalies in passenger count, coordinates, or duration?
- What candidate cleaning rules should we test before modeling?

Run from the project root:
    python scripts/01_data_exploration.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "reports" / "figures" / "01_exploration"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

train = pd.read_csv(RAW_DIR / "train.csv", parse_dates=["pickup_datetime", "dropoff_datetime"])
test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["pickup_datetime"])

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# ---------------------------------------------------------------------------
# Schema and missing values
# ---------------------------------------------------------------------------

schema = pd.DataFrame(
    {
        "train_dtype": train.dtypes,
        "train_missing": train.isna().sum(),
        "test_dtype": test.dtypes,
        "test_missing": test.isna().sum(),
    }
)
print("\n--- Schema ---")
print(schema.to_string())

# ---------------------------------------------------------------------------
# Target variable: trip_duration (seconds)
# ---------------------------------------------------------------------------

duration_summary = train["trip_duration"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
)
print("\n--- trip_duration summary ---")
print(duration_summary.to_frame().to_string())

train = train.assign(
    trip_duration_min=train["trip_duration"] / 60,
    log_trip_duration=np.log1p(train["trip_duration"]),
)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

upper_99 = train["trip_duration_min"].quantile(0.99)
sns.histplot(
    train.loc[train["trip_duration_min"] <= upper_99, "trip_duration_min"],
    bins=60,
    ax=axes[0],
)
axes[0].set_title("Trip duration, clipped at 99th percentile")
axes[0].set_xlabel("Duration (minutes)")

sns.histplot(train["log_trip_duration"], bins=60, ax=axes[1])
axes[1].set_title("Log-transformed trip duration")
axes[1].set_xlabel("log(1 + duration seconds)")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "trip_duration_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: trip_duration_distribution.png")

# ---------------------------------------------------------------------------
# Time patterns
# ---------------------------------------------------------------------------

train = train.assign(
    pickup_hour=train["pickup_datetime"].dt.hour,
    pickup_weekday=train["pickup_datetime"].dt.day_name(),
    pickup_month=train["pickup_datetime"].dt.month,
)

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=train, x="pickup_hour", ax=axes[0], color="#4C78A8")
axes[0].set_title("Trips by pickup hour")
axes[0].set_xlabel("Pickup hour")
axes[0].set_ylabel("Trips")

sns.countplot(data=train, x="pickup_weekday", order=weekday_order, ax=axes[1], color="#59A14F")
axes[1].set_title("Trips by weekday")
axes[1].set_xlabel("Pickup weekday")
axes[1].set_ylabel("Trips")
axes[1].tick_params(axis="x", rotation=35)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "trips_by_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: trips_by_time.png")

# ---------------------------------------------------------------------------
# Passenger count and store-and-forward flag
# ---------------------------------------------------------------------------

print("\n--- Passenger count distribution ---")
print(train["passenger_count"].value_counts().sort_index().to_frame("trips").to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=train, x="passenger_count", ax=axes[0], color="#F28E2B")
axes[0].set_title("Trips by passenger count")
axes[0].set_xlabel("Passenger count")
axes[0].set_ylabel("Trips")

sns.countplot(data=train, x="store_and_fwd_flag", ax=axes[1], color="#E15759")
axes[1].set_title("Store-and-forward flag")
axes[1].set_xlabel("Flag")
axes[1].set_ylabel("Trips")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "passenger_and_flag_counts.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: passenger_and_flag_counts.png")

# ---------------------------------------------------------------------------
# Location ranges
# ---------------------------------------------------------------------------

coord_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
print("\n--- Coordinate ranges ---")
print(train[coord_cols].agg(["min", "max"]).to_string())

NYC_BOUNDS = {"min_lon": -74.30, "max_lon": -73.60, "min_lat": 40.45, "max_lat": 41.00}

sample = train.sample(n=min(50_000, len(train)), random_state=42)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(sample["pickup_longitude"], sample["pickup_latitude"], s=1, alpha=0.15, label="Pickup")
ax.scatter(sample["dropoff_longitude"], sample["dropoff_latitude"], s=1, alpha=0.15, label="Dropoff")
ax.set_xlim(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
ax.set_ylim(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
ax.set_title("Sampled pickup and dropoff locations around NYC")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(markerscale=5)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "sampled_trip_locations.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: sampled_trip_locations.png")

# ---------------------------------------------------------------------------
# Candidate cleaning rules
# ---------------------------------------------------------------------------

duration_mask = train["trip_duration"].between(60, 6 * 60 * 60)
passenger_mask = train["passenger_count"].between(1, 6)
pickup_coord_mask = (
    train["pickup_longitude"].between(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
    & train["pickup_latitude"].between(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
)
dropoff_coord_mask = (
    train["dropoff_longitude"].between(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
    & train["dropoff_latitude"].between(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
)

quality_checks = pd.DataFrame(
    {
        "check": [
            "duration between 1 minute and 6 hours",
            "passenger count between 1 and 6",
            "pickup coordinate inside broad NYC bounds",
            "dropoff coordinate inside broad NYC bounds",
        ],
        "records_passing": [
            duration_mask.sum(),
            passenger_mask.sum(),
            pickup_coord_mask.sum(),
            dropoff_coord_mask.sum(),
        ],
    }
)
quality_checks["records_failing"] = len(train) - quality_checks["records_passing"]
quality_checks["share_failing"] = quality_checks["records_failing"] / len(train)

print("\n--- Quality checks ---")
print(quality_checks.to_string(index=False))

clean_mask = duration_mask & passenger_mask & pickup_coord_mask & dropoff_coord_mask

summary = pd.DataFrame(
    {
        "dataset": ["raw train", "after candidate filters"],
        "rows": [len(train), clean_mask.sum()],
        "share": [1.0, clean_mask.mean()],
    }
)
print("\n--- Filter summary ---")
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Initial findings
# ---------------------------------------------------------------------------
#
# - Regression problem: target is continuous (trip_duration in seconds).
# - trip_duration is right-skewed; log-transform likely needed for modeling.
# - Hour and weekday show clear volume patterns; both are candidate features.
# - Passenger count has rare zero/outlier values worth filtering.
# - Candidate filters remove a small fraction of rows (~1%) and look reasonable.
# - Next: turn these filters into a reusable cleaning step and add distance features.
