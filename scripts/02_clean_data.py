#!/usr/bin/env python3
"""
Clean the raw training data by removing invalid or unrealistic trips.

Cleaning rules (each documented with the rationale from exploration):

  1. Duration < 60 s     — too short to be a real metered trip; likely cancellations
                           or data entry errors.
  2. Duration > 6 h      — 99th percentile is ~57 min; trips beyond 6 h are almost
                           certainly GPS failures or data errors. The raw max is 41 days.
  3. Passenger count = 0 — no passengers recorded; ambiguous and rare (60 rows).
  4. Passenger count > 6 — exceeds NYC TLC legal maximum for a standard taxi.
  5. Pickup/dropoff outside broad NYC bounds — coordinates far outside the metro area
                           indicate GPS glitches. Bounds used: lon [-74.30, -73.60],
                           lat [40.45, 41.00].

These rules remove ~0.77 % of raw rows and are intentionally conservative. Tighter
thresholds can be tested during feature engineering.

Run from the project root:
    python3 scripts/02_clean_data.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
OUT_PATH = ROOT / "data" / "processed" / "train_clean.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

train = pd.read_csv(TRAIN_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])
n_raw = len(train)
print(f"Loaded {n_raw:,} rows from {TRAIN_PATH.name}")

# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------

NYC_BOUNDS = {"min_lon": -74.30, "max_lon": -73.60, "min_lat": 40.45, "max_lat": 41.00}

rules = {
    "duration_too_short":  train["trip_duration"] < 60,
    "duration_too_long":   train["trip_duration"] > 6 * 3600,
    "passenger_zero":      train["passenger_count"] == 0,
    "passenger_too_many":  train["passenger_count"] > 6,
    "pickup_outside_nyc":  ~(
        train["pickup_longitude"].between(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
        & train["pickup_latitude"].between(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
    ),
    "dropoff_outside_nyc": ~(
        train["dropoff_longitude"].between(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
        & train["dropoff_latitude"].between(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
    ),
}

# ---------------------------------------------------------------------------
# Report each rule individually before applying
# ---------------------------------------------------------------------------

print("\n--- Rows removed per rule ---")
for name, mask in rules.items():
    n = mask.sum()
    print(f"  {name:<25}  {n:>6,}  ({n / n_raw:.4%})")

# Combined mask: keep rows that fail none of the rules
invalid = pd.concat(rules.values(), axis=1).any(axis=1)
clean = train[~invalid].copy()

n_removed = n_raw - len(clean)
print(f"\nTotal removed : {n_removed:,} ({n_removed / n_raw:.4%})")
print(f"Rows remaining: {len(clean):,} ({len(clean) / n_raw:.4%})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

clean.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned dataset to {OUT_PATH}")
