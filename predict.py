#!/usr/bin/env python3
"""
Predict NYC taxi trip duration from pickup/dropoff coordinates and time.

Loads the best model trained by scripts/05_model_comparison.py and returns
an estimated trip duration in minutes.

Usage:
    python predict.py <pickup_lat> <pickup_lon> <dropoff_lat> <dropoff_lon> <hour> <weekday>

Arguments:
    pickup_lat   Pickup latitude  (e.g. 40.767)
    pickup_lon   Pickup longitude (e.g. -73.982)
    dropoff_lat  Dropoff latitude (e.g. 40.765)
    dropoff_lon  Dropoff longitude (e.g. -73.965)
    hour         Hour of pickup, 0–23
    weekday      Day of week, 0=Monday … 6=Sunday

Example:
    python predict.py 40.767 -73.982 40.765 -73.965 17 0

Run `make pipeline` first to train and save the model.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6_371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def predict(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, hour, weekday):
    """
    Predict trip duration in minutes.

    Parameters
    ----------
    pickup_lat, pickup_lon   : float  GPS coordinates of the pickup point
    dropoff_lat, dropoff_lon : float  GPS coordinates of the dropoff point
    hour                     : int    Hour of pickup (0–23)
    weekday                  : int    Day of week (0=Monday, 6=Sunday)

    Returns
    -------
    float  Predicted trip duration in minutes
    """
    model_path = MODELS_DIR / "best_model.joblib"
    features_path = MODELS_DIR / "features.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Run `make pipeline` first to train and save the model."
        )

    model    = joblib.load(model_path)
    features = joblib.load(features_path)

    is_weekend  = weekday >= 5
    is_rush     = (not is_weekend) and (hour in range(7, 9) or hour in range(17, 19))

    row = {
        "haversine_km":       haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon),
        "bearing_deg":        bearing_deg(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon),
        "pickup_hour":        hour,
        "pickup_weekday":     weekday,
        "pickup_month":       6,   # default: June (mid-dataset)
        "is_weekend":         float(is_weekend),
        "is_rush_hour":       float(is_rush),
        "passenger_count":    1,
        "vendor_id":          1,
        "store_and_fwd_flag": 0,
    }

    X = pd.DataFrame([row])[features]
    log_pred = model.predict(X)[0]
    return (np.expm1(log_pred)) / 60   # seconds → minutes


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)

    pickup_lat  = float(sys.argv[1])
    pickup_lon  = float(sys.argv[2])
    dropoff_lat = float(sys.argv[3])
    dropoff_lon = float(sys.argv[4])
    hour        = int(sys.argv[5])
    weekday     = int(sys.argv[6])

    minutes = predict(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, hour, weekday)
    dist    = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    day     = WEEKDAY_NAMES[weekday]

    print(f"\nTrip summary")
    print(f"  Distance  : {dist:.2f} km")
    print(f"  Departure : {day} at {hour:02d}:00")
    print(f"  Estimated : {minutes:.1f} minutes")


if __name__ == "__main__":
    main()
