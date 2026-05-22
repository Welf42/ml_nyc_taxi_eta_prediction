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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_features.csv"
FIGURES_DIR = ROOT / "figures" / "05_models"
MODELS_DIR = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BG    = "#0d1117"
FG    = "#F9FAFB"
FAINT = "#D1D5DB"
GRID  = "#1E2736"
CYAN  = "#22D3EE"
AMBER = "#F59E0B"
GREEN = "#10B981"
RED   = "#F87171"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GRID,
    "text.color":        FG,
    "axes.labelcolor":   FG,
    "xtick.color":       FAINT,
    "ytick.color":       FAINT,
    "axes.titlecolor":   FG,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#1E2736",
    "legend.edgecolor":  GRID,
    "legend.labelcolor": FG,
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

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(
    results_df["model"][::-1],
    results_df["val_RMSE"][::-1],
    color=CYAN,
    height=0.55,
    edgecolor=BG,
)
for bar, val in zip(bars, results_df["val_RMSE"][::-1]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=10, color=FG)
ax.set_xlabel("Validation RMSE  (log scale)")
ax.set_title("Model comparison — OLS to gradient boosting")
ax.set_xlim(0, results_df["val_RMSE"].max() * 1.15)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved: model_comparison.png")

# ---------------------------------------------------------------------------
# Feature importance — LightGBM (best model)
# ---------------------------------------------------------------------------

# Order features by absolute OLS coefficient — same ordering used in the coefficients plot
ols_abs_order = (
    pd.Series(ols.coef_, index=FEATURES)
    .sort_values(ascending=True)   # ascending → most negative first → top of barh = most positive
    .index
)
labels_ordered = [FEATURE_LABELS[f] for f in ols_abs_order]
lgbm_imp = pd.Series(lgbm.feature_importances_,
                     index=[FEATURE_LABELS[f] for f in FEATURES]).reindex(labels_ordered)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(lgbm_imp.index, lgbm_imp.values, color=GREEN, height=0.6, edgecolor=BG)
ax.set_xlim(left=0)
ax.set_title("LightGBM — feature importance")
ax.set_xlabel("Split gain")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "lgbm_split_gain.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: lgbm_split_gain.png")

# ---------------------------------------------------------------------------
# Permutation importance — LightGBM on validation set
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_val), size=min(10_000, len(X_val)), replace=False)
X_perm = X_val.iloc[sample_idx]
y_perm = y_val.iloc[sample_idx]

perm = permutation_importance(
    lgbm, X_perm, y_perm,
    n_repeats=5, random_state=42, n_jobs=-1,
    scoring="neg_root_mean_squared_error",
)
perm_imp = (
    pd.Series(perm.importances_mean, index=[FEATURE_LABELS[f] for f in FEATURES])
    .reindex(labels_ordered)
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(perm_imp.index, perm_imp.values, color=AMBER, height=0.6, edgecolor=BG)
ax.set_xlim(left=0)
ax.set_title("LightGBM — permutation importance")
ax.set_xlabel("Mean RMSE increase when feature is shuffled")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "lgbm_permutation.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: lgbm_permutation.png")

# ---------------------------------------------------------------------------
# OLS vs permutation importance — side-by-side comparison (both normalised 0–1)
# ---------------------------------------------------------------------------

ols_abs = (
    pd.Series(np.abs(ols.coef_), index=[FEATURE_LABELS[f] for f in FEATURES])
    .reindex(labels_ordered)
)
ols_norm  = ols_abs  / ols_abs.max()
perm_norm = perm_imp / perm_imp.max()

features  = ols_norm.index.tolist()
n         = len(features)
y         = np.arange(n)
height    = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(y + height / 2, ols_norm.values,  height=height, color=CYAN,  label="OLS  |coefficient|",    edgecolor=BG)
ax.barh(y - height / 2, perm_norm.values, height=height, color=AMBER, label="Permutation importance", edgecolor=BG)

ax.set_yticks(y)
ax.set_yticklabels(features)
ax.set_xlim(left=0)
ax.set_xlabel("Normalised importance  (0 = least, 1 = most)")
ax.set_title("OLS vs LightGBM — what each method sees")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ols_vs_lgbm.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: ols_vs_lgbm.png")

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
                   "pickup_longitude", "pickup_latitude",
                   "trip_duration", "log_trip_duration"]].copy()
val_out["predicted_log"] = best["pred"]
val_out["residual"]      = val_out["log_trip_duration"] - val_out["predicted_log"]
val_out.to_csv(ROOT / "data" / "processed" / "val_predictions.csv", index=False)
print("Saved: data/processed/val_predictions.csv  (input for 06_error_analysis.py)")
