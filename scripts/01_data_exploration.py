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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from styles import BG, FG, FAINT, GRID, CYAN, AMBER, GREEN, RED, apply_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "figures" / "01_exploration"
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

pd.set_option("display.max_columns", 50)

apply_theme()

upper_99 = train["trip_duration_min"].quantile(0.99)
median_min = train["trip_duration_min"].median()

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
    train.loc[train["trip_duration_min"] <= upper_99, "trip_duration_min"],
    bins=60, ax=ax, color=CYAN, edgecolor=BG, linewidth=0.3,
)
ax.axvline(median_min, color=RED, linewidth=1.5, linestyle="--")
ax.text(median_min + 0.5, ax.get_ylim()[1] * 0.92,
        f"Median: {median_min:.0f} min", color=RED, fontsize=9)
ax.set_title("Trip duration — raw")
ax.set_xlabel("Duration (minutes)  —  clipped at 99th percentile")
ax.set_ylabel("Trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "trips_by_duration_raw.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: trips_by_duration_raw.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(train["log_trip_duration"], bins=60, ax=ax, color=GREEN, edgecolor=BG, linewidth=0.3)
ax.set_title("Trip duration — log scale")
ax.set_xlabel("log(1 + duration in seconds)")
ax.set_ylabel("Trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "trips_by_duration_log.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: trips_by_duration_log.png")

# ---------------------------------------------------------------------------
# Time patterns
# ---------------------------------------------------------------------------

train = train.assign(
    pickup_hour=train["pickup_datetime"].dt.hour,
    pickup_weekday=train["pickup_datetime"].dt.day_name(),
    pickup_month=train["pickup_datetime"].dt.month,
)

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=train, x="pickup_hour", ax=ax, color=CYAN)
ax.set_title("Trips by pickup hour")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "trips_by_hour.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: trips_by_hour.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=train, x="pickup_weekday", order=weekday_order, ax=ax, color=AMBER)
ax.set_title("Trips by weekday")
ax.set_xlabel("")
ax.set_ylabel("Trips")
ax.tick_params(axis="x", rotation=35)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "trips_by_weekday.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: trips_by_weekday.png")

# ---------------------------------------------------------------------------
# Passenger count and store-and-forward flag
# ---------------------------------------------------------------------------

print("\n--- Passenger count distribution ---")
print(train["passenger_count"].value_counts().sort_index().to_frame("trips").to_string())

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=train, x="passenger_count", ax=ax, color=GREEN)
ax.set_title("Trips by passenger count")
ax.set_xlabel("Passenger count")
ax.set_ylabel("Trips")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "trips_by_passengers.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: trips_by_passengers.png")

# ---------------------------------------------------------------------------
# Location ranges
# ---------------------------------------------------------------------------

coord_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
print("\n--- Coordinate ranges ---")
print(train[coord_cols].agg(["min", "max"]).to_string())

NYC_BOUNDS = {"min_lon": -74.30, "max_lon": -73.60, "min_lat": 40.45, "max_lat": 41.00}

sample = train.sample(n=min(50_000, len(train)), random_state=42)

fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
ax.set_facecolor(BG)

ax.scatter(sample["pickup_longitude"], sample["pickup_latitude"],
           s=0.4, alpha=0.35, color="#00B4D8", label="Pickup", rasterized=True)
ax.scatter(sample["dropoff_longitude"], sample["dropoff_latitude"],
           s=0.4, alpha=0.35, color="#FF6B6B", label="Dropoff", rasterized=True)

ax.set_xlim(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
ax.set_ylim(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
ax.set_title("Pickup and dropoff locations — NYC", color="white", pad=14, fontsize=14, fontweight="bold")
ax.set_xlabel("Longitude", color="#888888", fontsize=10)
ax.set_ylabel("Latitude",  color="#888888", fontsize=10)
ax.tick_params(colors="#555555", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("#222222")

ax.legend(markerscale=10, fontsize=10,
          facecolor="#1E2736", edgecolor=GRID, labelcolor=FG)
ax.text(0.02, 0.02, f"{len(sample):,} sampled trips",
        transform=ax.transAxes, color="#555555", fontsize=9, va="bottom")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "trips_by_location.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: trips_by_location.png")

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
