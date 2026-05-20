#!/usr/bin/env python3
"""
Train and compare three regression models for NYC taxi trip duration.

Models (all from scikit-learn):

  OLS — Ordinary Least Squares
    Fits a straight line through the data. Fast, interpretable, sensitive to
    non-linear relationships. Used as the statistical baseline.

  Random Forest
    Builds many decision trees on random subsets of data and features, then
    averages their predictions. Handles non-linearity and feature interactions
    without manual feature engineering. Less sensitive to outliers than OLS.

  Histogram Gradient Boosting (HistGBR)
    Builds trees sequentially, each one correcting the errors of the previous.
    Sklearn's fast boosting implementation (similar to LightGBM internally).
    Often the strongest out-of-the-box model for tabular regression.

All models use the same time-ordered 80/20 train/validation split and the
same 10 engineered features. Metric: RMSE on log_trip_duration (= RMSLE).

Run from the project root:
    python3 scripts/05_model_comparison.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_features.csv"
FIGURES_DIR = ROOT / "reports" / "figures" / "05_comparison"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Load and split (time-ordered 80/20)
# ---------------------------------------------------------------------------

df = pd.read_csv(IN_PATH, parse_dates=["pickup_datetime"])
df = df.sort_values("pickup_datetime").reset_index(drop=True)

FEATURES = [
    "haversine_km",
    "bearing_deg",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "passenger_count",
    "vendor_id",
    "store_and_fwd_flag",
]
TARGET = "log_trip_duration"

split = int(len(df) * 0.8)
train_df, val_df = df.iloc[:split], df.iloc[split:]

X_train = train_df[FEATURES].astype(float)
y_train = train_df[TARGET]
X_val   = val_df[FEATURES].astype(float)
y_val   = val_df[TARGET]

print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

results = []

# OLS
t0 = time.time()
ols = LinearRegression().fit(X_train_sc, y_train)
ols_pred = ols.predict(X_val_sc)
results.append({
    "model":    "OLS",
    "val_RMSE": root_mean_squared_error(y_val, ols_pred),
    "train_s":  round(time.time() - t0, 1),
    "pred":     ols_pred,
})
print(f"OLS              RMSE={results[-1]['val_RMSE']:.4f}  ({results[-1]['train_s']}s)")

# Random Forest (capped trees for speed; tune n_estimators for production)
t0 = time.time()
rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
results.append({
    "model":    "Random Forest",
    "val_RMSE": root_mean_squared_error(y_val, rf_pred),
    "train_s":  round(time.time() - t0, 1),
    "pred":     rf_pred,
})
print(f"Random Forest    RMSE={results[-1]['val_RMSE']:.4f}  ({results[-1]['train_s']}s)")

# Histogram Gradient Boosting
t0 = time.time()
hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
hgb.fit(X_train, y_train)
hgb_pred = hgb.predict(X_val)
results.append({
    "model":    "Gradient Boosting",
    "val_RMSE": root_mean_squared_error(y_val, hgb_pred),
    "train_s":  round(time.time() - t0, 1),
    "pred":     hgb_pred,
})
print(f"Gradient Boosting RMSE={results[-1]['val_RMSE']:.4f}  ({results[-1]['train_s']}s)")

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in results])
results_df = results_df.sort_values("val_RMSE").reset_index(drop=True)

print("\n--- Results ---")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# RMSE comparison chart
# ---------------------------------------------------------------------------

colors = ["#4C78A8", "#59A14F", "#F28E2B"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(
    results_df["model"][::-1],
    results_df["val_RMSE"][::-1],
    color=colors[:len(results_df)][::-1],
    height=0.5,
    edgecolor="white",
)
for bar, val in zip(bars, results_df["val_RMSE"][::-1]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=10, color="#333333")
ax.set_xlabel("Validation RMSE  (log_trip_duration)")
ax.set_title("Model comparison")
ax.set_xlim(0, results_df["val_RMSE"].max() * 1.15)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: model_comparison.png")

# ---------------------------------------------------------------------------
# Feature importance — Random Forest and Gradient Boosting
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "haversine_km":       "Distance (km)",
    "bearing_deg":        "Direction (bearing)",
    "pickup_hour":        "Pickup hour",
    "pickup_weekday":     "Weekday",
    "pickup_month":       "Month",
    "is_weekend":         "Is weekend",
    "is_rush_hour":       "Rush hour",
    "passenger_count":    "Passenger count",
    "vendor_id":          "Vendor",
    "store_and_fwd_flag": "Store & forward",
}
labels = [FEATURE_LABELS[f] for f in FEATURES]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest exposes feature_importances_ directly (mean decrease in impurity)
rf_imp = pd.Series(rf.feature_importances_, index=labels).sort_values()
axes[0].barh(rf_imp.index, rf_imp.values, color="#59A14F", height=0.6, edgecolor="white")
axes[0].set_title("Random Forest — feature importance")
axes[0].set_xlabel("Mean decrease in impurity")

# HistGBR requires permutation importance (sample for speed)
sample_idx = np.random.default_rng(42).choice(len(X_val), size=10_000, replace=False)
perm = permutation_importance(
    hgb, X_val.iloc[sample_idx], y_val.iloc[sample_idx],
    n_repeats=5, random_state=42, n_jobs=-1,
)
hgb_imp = pd.Series(perm.importances_mean, index=labels).sort_values()
axes[1].barh(hgb_imp.index, hgb_imp.values, color="#F28E2B", height=0.6, edgecolor="white")
axes[1].set_title("Gradient Boosting — feature importance\n(permutation)")
axes[1].set_xlabel("Mean RMSE increase when feature is shuffled")

fig.suptitle("Which features drive each model?", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: feature_importance.png")

# ---------------------------------------------------------------------------
# Save best model predictions for error analysis in script 06
# ---------------------------------------------------------------------------

best = min(results, key=lambda r: r["val_RMSE"])
print(f"\nBest model: {best['model']}  (RMSE {best['val_RMSE']:.4f})")

val_out = val_df[["id", "pickup_datetime", "haversine_km",
                   "pickup_hour", "pickup_weekday", "is_weekend",
                   "trip_duration", "log_trip_duration"]].copy()
val_out["predicted_log"] = best["pred"]
val_out["residual"]      = val_out["log_trip_duration"] - val_out["predicted_log"]
val_out.to_csv(ROOT / "data" / "processed" / "val_predictions.csv", index=False)
print("Saved: data/processed/val_predictions.csv  (input for 06_error_analysis.py)")
