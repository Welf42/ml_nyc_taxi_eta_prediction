#!/usr/bin/env python3
"""
Train and compare all regression models for NYC taxi trip duration.

Progression from simple to complex:

  OLS                  — linear baseline; interpretable coefficients
  Random Forest        — ensemble of trees; handles non-linearity
  Hist. Grad. Boost    — sklearn's fast gradient boosting
  LightGBM             — leaf-wise boosting; fast and accurate
  XGBoost              — level-wise boosting with built-in regularisation
  LightGBM (tuned)     — Optuna hyperparameter search, 20 trials

All models use the same time-ordered 80/20 train/validation split and the
same 10 engineered features. Metric: RMSE on log_trip_duration (= RMSLE).

After tuning, SHAP values are computed on 5 000 validation samples to show
which features drive predictions and in which direction.

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
import optuna
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from styles import BG, FG, FAINT, GRID, CYAN, AMBER, GREEN, RED, apply_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_features.csv"
FIGURES_DIR = ROOT / "figures" / "05_models"
MODELS_DIR = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()

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

# ---------------------------------------------------------------------------
# Optuna hyperparameter search — LightGBM
# ---------------------------------------------------------------------------
# Searches 20 configurations; uses early stopping within each trial to keep
# runtime reasonable. For strict methodology, replace the val set here with
# cross-validation folds to avoid tuning on the test set.

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _lgbm_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "n_jobs": -1, "random_state": 42, "verbose": -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return root_mean_squared_error(y_val, m.predict(X_val))

print("\nTuning LightGBM with Optuna (20 trials)...")
t0 = time.time()
study = optuna.create_study(direction="minimize")
study.optimize(_lgbm_objective, n_trials=20, show_progress_bar=False)
elapsed_tune = time.time() - t0

best_params = study.best_params | {"n_jobs": -1, "random_state": 42, "verbose": -1}
lgbm_tuned = lgb.LGBMRegressor(**best_params)
lgbm_tuned.fit(X_train, y_train)
record("LightGBM (tuned)", "lightgbm+optuna", lgbm_tuned, lgbm_tuned.predict(X_val), elapsed_tune)
print(f"  Best params: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in study.best_params.items()} }")

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
# SHAP — explain the tuned LightGBM
# ---------------------------------------------------------------------------
# TreeExplainer is exact (not sampled) for tree models and runs in seconds.
# 5 000 rows is enough for a stable beeswarm; full val set would take longer.

print("\nComputing SHAP values (5 000 validation samples)...")
rng_shap = np.random.default_rng(0)
shap_idx = rng_shap.choice(len(X_val), size=min(5_000, len(X_val)), replace=False)
X_shap = X_val.iloc[shap_idx].rename(columns=FEATURE_LABELS)

explainer = shap.TreeExplainer(lgbm_tuned)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(shap_values, X_shap, show=False, plot_size=(10, 6))
fig = plt.gcf()
ax  = plt.gca()

# SHAP draws its own text elements that bypass rcParams — repaint them all.
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.tick_params(colors=FAINT, labelcolor=FAINT)
ax.xaxis.label.set_color(FAINT)
for text in ax.texts:
    text.set_color(FG)
for text in fig.texts:
    text.set_color(FG)
# y-axis tick labels (feature names) are the most visible dark elements
for lbl in ax.get_yticklabels():
    lbl.set_color(FG)
for lbl in ax.get_xticklabels():
    lbl.set_color(FAINT)
# colorbar
for child in fig.get_children():
    if hasattr(child, "yaxis"):
        child.set_facecolor(BG)
        child.tick_params(colors=FAINT, labelcolor=FAINT)
        child.yaxis.label.set_color(FAINT)

ax.set_title("LightGBM (tuned) — SHAP feature impact",
             color=FG, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: shap_summary.png")

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
