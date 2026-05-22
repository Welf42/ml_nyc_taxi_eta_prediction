#!/usr/bin/env python3
"""
Train and compare all regression models for NYC taxi trip duration.

Progression from simple to complex:

  OLS                  — linear baseline; interpretable coefficients
  Random Forest        — ensemble of trees; handles non-linearity
  Hist. Grad. Boost    — sklearn's fast gradient boosting
  LightGBM             — leaf-wise boosting; fast and accurate
  XGBoost              — level-wise boosting with built-in regularisation

All models use the same time-ordered 80/20 train/validation split and the
same 10 engineered features. Metric: RMSE on log_trip_duration (= RMSLE).

The best model is saved to models/ for use by predict.py.

Run from the project root:
    python3 scripts/05_model_comparison.py
"""

import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_features.csv"
FIGURES_DIR = ROOT / "reports" / "figures" / "05_model_comparison"
MODELS_DIR = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

# ---------------------------------------------------------------------------
# Load and split (time-ordered 80/20)
# ---------------------------------------------------------------------------

df = pd.read_csv(IN_PATH, parse_dates=["pickup_datetime"])
df = df.sort_values("pickup_datetime").reset_index(drop=True)

FEATURES = list(FEATURE_LABELS.keys())
TARGET = "log_trip_duration"

split = int(len(df) * 0.8)
train_df, val_df = df.iloc[:split], df.iloc[split:]

X_train = train_df[FEATURES].astype(float)
y_train = train_df[TARGET]
X_val   = val_df[FEATURES].astype(float)
y_val   = val_df[TARGET]

print(f"Train: {len(train_df):,}  ({train_df['pickup_datetime'].min().date()} – {train_df['pickup_datetime'].max().date()})")
print(f"Val  : {len(val_df):,}  ({val_df['pickup_datetime'].min().date()} – {val_df['pickup_datetime'].max().date()})")

# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

results = []
trained = {}

def record(name, library, model, pred, elapsed, needs_scale=False):
    rmse = root_mean_squared_error(y_val, pred)
    results.append({"model": name, "library": library, "val_RMSE": rmse, "train_s": round(elapsed, 1)})
    trained[name] = {"model": model, "pred": pred, "needs_scale": needs_scale}
    print(f"  {name:<26}  RMSE={rmse:.4f}  ({elapsed:.1f}s)")

print("\nTraining models...")

t0 = time.time()
ols = LinearRegression().fit(X_train_sc, y_train)
record("OLS", "scikit-learn", ols, ols.predict(X_val_sc), time.time() - t0, needs_scale=True)

t0 = time.time()
rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
record("Random Forest", "scikit-learn", rf, rf.predict(X_val), time.time() - t0)

t0 = time.time()
hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
hgb.fit(X_train, y_train)
record("Hist. Grad. Boost", "scikit-learn", hgb, hgb.predict(X_val), time.time() - t0)

t0 = time.time()
lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          num_leaves=63, n_jobs=-1, random_state=42, verbose=-1)
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
record("LightGBM", "lightgbm", lgbm, lgbm.predict(X_val), time.time() - t0)

t0 = time.time()
xgbm = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                          random_state=42, verbosity=0)
xgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
record("XGBoost", "xgboost", xgbm, xgbm.predict(X_val), time.time() - t0)

results_df = pd.DataFrame(results).sort_values("val_RMSE").reset_index(drop=True)

print("\n--- Results ---")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# RMSE comparison chart
# ---------------------------------------------------------------------------

palette = {"scikit-learn": "#4C78A8", "lightgbm": "#59A14F", "xgboost": "#F28E2B"}
bar_colors = [palette[lib] for lib in results_df["library"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(
    results_df["model"][::-1],
    results_df["val_RMSE"][::-1],
    color=bar_colors[::-1],
    height=0.55,
    edgecolor="white",
)
for bar, val in zip(bars, results_df["val_RMSE"][::-1]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=10, color="#333333")
ax.set_xlabel("Validation RMSE  (log_trip_duration)")
ax.set_title("Model comparison — OLS to gradient boosting")
ax.set_xlim(0, results_df["val_RMSE"].max() * 1.15)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=lib) for lib, c in palette.items()
                   if lib in results_df["library"].values]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: model_comparison.png")

# ---------------------------------------------------------------------------
# Feature importance — LightGBM (best model)
# ---------------------------------------------------------------------------

labels = [FEATURE_LABELS[f] for f in FEATURES]
lgbm_imp = pd.Series(lgbm.feature_importances_, index=labels).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(lgbm_imp.index, lgbm_imp.values, color="#59A14F", height=0.6, edgecolor="white")
ax.set_title("LightGBM — feature importance")
ax.set_xlabel("Split gain")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: feature_importance.png")

# ---------------------------------------------------------------------------
# Save best model and val predictions for downstream scripts
# ---------------------------------------------------------------------------

best_name = results_df.iloc[0]["model"]
best = trained[best_name]
print(f"\nBest model: {best_name}  (RMSE {results_df.iloc[0]['val_RMSE']:.4f})")

joblib.dump(best["model"], MODELS_DIR / "best_model.joblib")
joblib.dump(FEATURES, MODELS_DIR / "features.joblib")
print(f"Saved: models/best_model.joblib  ({best_name})")

val_out = val_df[["id", "pickup_datetime", "haversine_km",
                   "pickup_hour", "pickup_weekday", "is_weekend",
                   "trip_duration", "log_trip_duration"]].copy()
val_out["predicted_log"] = best["pred"]
val_out["residual"]      = val_out["log_trip_duration"] - val_out["predicted_log"]
val_out.to_csv(ROOT / "data" / "processed" / "val_predictions.csv", index=False)
print("Saved: data/processed/val_predictions.csv  (input for 06_error_analysis.py)")
