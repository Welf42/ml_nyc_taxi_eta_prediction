#!/usr/bin/env python3
"""
Advanced model comparison: LightGBM and XGBoost vs. the sklearn baseline.

Theory notes (for presentation):

  LightGBM — Light Gradient Boosting Machine
    Gradient boosting framework by Microsoft. Grows trees leaf-wise rather than
    level-wise, which finds better splits faster. Uses histogram-based binning
    internally. Typically faster to train and slightly more accurate than
    sklearn's HistGradientBoostingRegressor on large tabular datasets.

  XGBoost — Extreme Gradient Boosting
    Gradient boosting framework by the DMLC group. Grows trees level-wise.
    Adds L1 and L2 regularisation terms directly into the objective, making it
    more robust on noisy features. The dominant Kaggle competition model for
    tabular data for many years.

  Both vs. sklearn HistGBR
    All three implement the same core algorithm (gradient boosting on decision
    trees), but LightGBM and XGBoost have more tuning options, better GPU
    support, and are faster at scale. For this dataset the differences will
    be small.

Metric: RMSE on log_trip_duration. Same 80/20 time-ordered split as scripts
04 and 05 so all results are directly comparable.

Run from the project root:
    python3 scripts/06_model_advanced.py
"""

import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_features.csv"
FIGURES_DIR = ROOT / "reports" / "figures" / "06_advanced"
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

print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

# ---------------------------------------------------------------------------
# Reference scores from script 05 (sklearn models, same split)
# ---------------------------------------------------------------------------

reference = [
    {"model": "OLS",               "val_RMSE": 0.5514, "library": "scikit-learn"},
    {"model": "Random Forest",     "val_RMSE": 0.3839, "library": "scikit-learn"},
    {"model": "Hist. Grad. Boost", "val_RMSE": 0.3778, "library": "scikit-learn"},
]

results = []

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_pred = lgb_model.predict(X_val)
lgb_rmse = root_mean_squared_error(y_val, lgb_pred)
results.append({"model": "LightGBM", "val_RMSE": lgb_rmse,
                "train_s": round(time.time() - t0, 1), "library": "lightgbm"})
print(f"LightGBM   RMSE={lgb_rmse:.4f}  ({results[-1]['train_s']}s)")

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb_model.predict(X_val)
xgb_rmse = root_mean_squared_error(y_val, xgb_pred)
results.append({"model": "XGBoost", "val_RMSE": xgb_rmse,
                "train_s": round(time.time() - t0, 1), "library": "xgboost"})
print(f"XGBoost    RMSE={xgb_rmse:.4f}  ({results[-1]['train_s']}s)")

# ---------------------------------------------------------------------------
# Full comparison table
# ---------------------------------------------------------------------------

all_results = pd.DataFrame(reference + results).sort_values("val_RMSE").reset_index(drop=True)
print("\n--- Full comparison ---")
print(all_results[["model", "library", "val_RMSE", "train_s"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Comparison chart (all models)
# ---------------------------------------------------------------------------

palette = {
    "scikit-learn": "#4C78A8",
    "lightgbm":     "#59A14F",
    "xgboost":      "#F28E2B",
}
bar_colors = [palette[lib] for lib in all_results["library"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(
    all_results["model"][::-1],
    all_results["val_RMSE"][::-1],
    color=bar_colors[::-1],
    height=0.55,
    edgecolor="white",
)
for bar, val in zip(bars, all_results["val_RMSE"][::-1]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=10, color="#333333")
ax.set_xlabel("Validation RMSE  (log_trip_duration)")
ax.set_title("All models compared")
ax.set_xlim(0, all_results["val_RMSE"].max() * 1.15)

# Legend by library
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=lib) for lib, c in palette.items()
                   if lib in all_results["library"].values]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "model_comparison_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: model_comparison_full.png")

# ---------------------------------------------------------------------------
# Feature importance — LightGBM and XGBoost
# ---------------------------------------------------------------------------

labels = [FEATURE_LABELS[f] for f in FEATURES]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

lgb_imp = pd.Series(lgb_model.feature_importances_, index=labels).sort_values()
axes[0].barh(lgb_imp.index, lgb_imp.values, color="#59A14F", height=0.6, edgecolor="white")
axes[0].set_title("LightGBM — feature importance")
axes[0].set_xlabel("Split gain")

xgb_imp = pd.Series(xgb_model.feature_importances_, index=labels).sort_values()
axes[1].barh(xgb_imp.index, xgb_imp.values, color="#F28E2B", height=0.6, edgecolor="white")
axes[1].set_title("XGBoost — feature importance")
axes[1].set_xlabel("F-score")

fig.suptitle("Which features drive each model?", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "feature_importance_advanced.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: feature_importance_advanced.png")

# ---------------------------------------------------------------------------
# Save best model predictions for error analysis (script 07)
# ---------------------------------------------------------------------------

best = all_results.iloc[0]
best_pred = lgb_pred if best["model"] == "LightGBM" else xgb_pred
print(f"\nBest model: {best['model']}  (RMSE {best['val_RMSE']:.4f})")

val_out = val_df[["id", "pickup_datetime", "haversine_km",
                   "pickup_hour", "pickup_weekday", "is_weekend",
                   "trip_duration", "log_trip_duration"]].copy()
val_out["predicted_log"] = best_pred
val_out["residual"]      = val_out["log_trip_duration"] - val_out["predicted_log"]
val_out.to_csv(ROOT / "data" / "processed" / "val_predictions.csv", index=False)
print("Saved: data/processed/val_predictions.csv  (input for 07_error_analysis.py)")
