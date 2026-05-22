#!/usr/bin/env python3
"""
Baseline model: Ordinary Least Squares regression with RMSE evaluation.

Theory notes (for presentation):

  OLS — Ordinary Least Squares
  ─────────────────────────────
  OLS fits a line (or hyperplane) through the data by minimising the sum of
  squared residuals:  min Σ (y_i − ŷ_i)²
  Each coefficient tells you how much log(duration) changes for a one-unit
  change in that feature, holding everything else constant.
  We standardise features first so coefficients are comparable across features
  with different scales (kilometres vs. hour-of-day vs. binary flags).

  RMSE — Root Mean Squared Error
  ────────────────────────────────
  RMSE = √[ (1/n) Σ (y_i − ŷ_i)² ]
  Because the target is log(1 + duration), an RMSE of X means predictions are
  off by roughly ±(e^X − 1) × 100% on average. For example:
    RMSE 0.55  ≈ ±73% error on raw seconds
    RMSE 0.48  ≈ ±62% error on raw seconds
  This is also the Kaggle competition metric (RMSLE on the raw target).

  Why log-transform the target?
  ──────────────────────────────
  trip_duration is right-skewed (a few very long trips). Squared errors on the
  raw scale would be dominated by those outliers, making the model optimise for
  rare cases. The log transform compresses the tail and makes residuals more
  symmetric — so RMSE becomes a meaningful average across all trips.

Steps:
  1. Naive baseline — predict the training mean for every trip (the floor).
  2. OLS regression — fit on 10 engineered features.
  3. Coefficient table — which features matter and in which direction?
  4. Residual plots   — are errors random, or is there structure left to model?

Train/validation split: 80/20, time-ordered (Jan–May → Jun) to simulate
predicting future trips, which is what the model would do in production.

Run from the project root:
    python3 scripts/04_baseline_model.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from styles import BG, FG, FAINT, GRID, CYAN, AMBER, GREEN, RED, apply_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_features.csv"
FIGURES_DIR = ROOT / "figures" / "04_baseline"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_theme()

# ---------------------------------------------------------------------------
# Load and sort by time
# ---------------------------------------------------------------------------

df = pd.read_csv(IN_PATH, parse_dates=["pickup_datetime"])
df = df.sort_values("pickup_datetime").reset_index(drop=True)
print(f"Loaded {len(df):,} rows")

FEATURES = [
    "haversine_km",       # straight-line distance between pickup and dropoff
    "bearing_deg",        # compass direction of the trip
    "pickup_hour",        # 0–23
    "pickup_weekday",     # 0 (Mon) – 6 (Sun)
    "pickup_month",       # 1–6
    "is_weekend",         # True/False
    "is_rush_hour",       # True/False (07–09 h and 17–19 h on weekdays)
    "passenger_count",
    "vendor_id",
    "store_and_fwd_flag",
]
TARGET = "log_trip_duration"

# ---------------------------------------------------------------------------
# Train / validation split — time-ordered, 80/20
# ---------------------------------------------------------------------------

split = int(len(df) * 0.8)
train_df, val_df = df.iloc[:split], df.iloc[split:]

print(f"\nTrain: {len(train_df):,} rows  ({train_df['pickup_datetime'].min().date()} – {train_df['pickup_datetime'].max().date()})")
print(f"Val  : {len(val_df):,} rows  ({val_df['pickup_datetime'].min().date()} – {val_df['pickup_datetime'].max().date()})")

X_train = train_df[FEATURES].astype(float)
y_train = train_df[TARGET]
X_val   = val_df[FEATURES].astype(float)
y_val   = val_df[TARGET]

# ---------------------------------------------------------------------------
# Step 1 — Naive baseline
# ---------------------------------------------------------------------------

naive_pred = np.full(len(y_val), y_train.mean())
naive_rmse = root_mean_squared_error(y_val, naive_pred)

print(f"\n--- Naive baseline ---")
print(f"Training mean (log scale): {y_train.mean():.4f}")
print(f"Validation RMSE          : {naive_rmse:.4f}")

# ---------------------------------------------------------------------------
# Step 2 — OLS regression
# ---------------------------------------------------------------------------

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

ols = LinearRegression()
ols.fit(X_train_sc, y_train)

ols_pred = ols.predict(X_val_sc)
ols_rmse = root_mean_squared_error(y_val, ols_pred)

print(f"\n--- OLS regression ---")
print(f"Validation RMSE    : {ols_rmse:.4f}")
print(f"Improvement vs naive: {(naive_rmse - ols_rmse) / naive_rmse:.1%}")
print(f"R²                 : {ols.score(X_val_sc, y_val):.4f}")

# ---------------------------------------------------------------------------
# Step 3 — Coefficients
# ---------------------------------------------------------------------------

coef_df = (
    pd.Series(ols.coef_, index=FEATURES, name="coefficient")
    .sort_values(ascending=False)
    .to_frame()
)
coef_df["direction"] = coef_df["coefficient"].apply(lambda x: "longer trip" if x > 0 else "shorter trip")

print(f"\n--- Coefficients (standardised features) ---")
print(coef_df.to_string())

FEATURE_LABELS = {
    "haversine_km":       "Distance (km)",
    "is_weekend":         "Is weekend",
    "pickup_hour":        "Pickup hour",
    "pickup_weekday":     "Weekday",
    "pickup_month":       "Month",
    "is_rush_hour":       "Rush hour",
    "passenger_count":    "Passenger count",
    "bearing_deg":        "Direction (bearing)",
    "vendor_id":          "Vendor",
    "store_and_fwd_flag": "Store & forward",
}
coef_plot = coef_df.copy()
coef_plot.index = [FEATURE_LABELS.get(i, i) for i in coef_plot.index]

labels_rev = coef_plot.index[::-1].tolist()
vals_rev   = coef_plot["coefficient"][::-1].tolist()
colors_rev = [CYAN if v >= 0 else RED for v in vals_rev]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels_rev, vals_rev, color=colors_rev, height=0.6, edgecolor="white")

for bar, val in zip(bars, vals_rev):
    if val >= 0:
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left", fontsize=9, color=FG)
    else:
        ax.text(val - 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="right", fontsize=9, color=FG)

# Pin spine to physical left edge so tick labels land in the figure margin, not inside the axes
ax.spines["left"].set_position(("axes", 0))
ax.set_xlim(-0.15, max(vals_rev) * 1.22)
ax.axvline(0, color=FAINT, linewidth=0.8)
ax.set_title("OLS regression coefficients")
ax.set_xlabel("Standardised coefficient  (cyan = positive, red = negative)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ols_coefficients.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved: ols_coefficients.png")

# ---------------------------------------------------------------------------
# Step 4 — Residual plots
# ---------------------------------------------------------------------------

residuals = y_val - ols_pred

# 1 — Residual distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(residuals, bins=80, ax=ax, color=CYAN, edgecolor=BG, linewidth=0.3)
ax.axvline(0, color=RED, linewidth=1.5, linestyle="--")
ax.set_title("OLS — residual distribution")
ax.set_xlabel("Actual − Predicted  (log scale)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ols_residuals_distribution.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: ols_residuals_distribution.png")

# 2 — Residuals vs. predicted
fig, ax = plt.subplots(figsize=(8, 5))
hb = ax.hexbin(ols_pred, residuals, gridsize=60, cmap="YlGnBu", mincnt=1, linewidths=0.1)
ax.axhline(0, color=RED, linewidth=1.5, linestyle="--")
ax.set_title("OLS — residuals vs. predicted")
ax.set_xlabel("Predicted  (log scale)")
ax.set_ylabel("Residual")
fig.colorbar(hb, ax=ax, label="Trips")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ols_residuals_vs_predicted.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: ols_residuals_vs_predicted.png")

# 3 — Actual vs. predicted
fig, ax = plt.subplots(figsize=(8, 5))
lims = [y_val.min(), y_val.max()]
hb2 = ax.hexbin(y_val, ols_pred, gridsize=60, cmap="plasma", mincnt=1, linewidths=0.1)
ax.plot(lims, lims, color=RED, linewidth=1.5, linestyle="--", label="Perfect prediction")
ax.set_title("OLS — actual vs. predicted")
ax.set_xlabel("Actual  (log scale)")
ax.set_ylabel("Predicted  (log scale)")
ax.legend(fontsize=9)
fig.colorbar(hb2, ax=ax, label="Trips")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ols_actual_vs_predicted.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved: ols_actual_vs_predicted.png")

# ---------------------------------------------------------------------------
# Summary table + RMSE comparison chart
# ---------------------------------------------------------------------------

models_summary = [
    {"model": "Naive (predict mean)", "val_RMSE": naive_rmse},
    {"model": "OLS regression",       "val_RMSE": ols_rmse},
]
summary = pd.DataFrame(models_summary)

print(f"\n--- Summary ---")
print(summary.round(4).to_string(index=False))
print(f"\nInterpretation: OLS RMSE of {ols_rmse:.4f} on log scale ≈ ±{(np.exp(ols_rmse) - 1) * 100:.0f}% error on raw trip duration.")

