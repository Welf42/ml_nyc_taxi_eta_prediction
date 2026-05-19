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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "train_features.csv"
FIGURES_DIR = ROOT / "reports" / "figures" / "04_baseline"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

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
    .sort_values(key=abs, ascending=False)
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

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#4C78A8" if v >= 0 else "#E15759" for v in coef_plot["coefficient"]]
bars = ax.barh(
    coef_plot.index[::-1],
    coef_plot["coefficient"][::-1],
    color=colors[::-1],
    height=0.6,
    edgecolor="white",
)
for bar, val in zip(bars, coef_plot["coefficient"][::-1]):
    offset = 0.005 if val >= 0 else -0.005
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=9, color="#333333")
ax.axvline(0, color="#333333", linewidth=0.8)
ax.set_title("What predicts a longer trip?")
ax.set_xlabel("OLS coefficient (standardised features)")
ax.text(0.98, 0.02, "Blue = longer  ·  Red = shorter",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#666666")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ols_coefficients.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: ols_coefficients.png")

# ---------------------------------------------------------------------------
# Step 4 — Residual plots
# ---------------------------------------------------------------------------

residuals = y_val - ols_pred

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1 — Residual distribution
sns.histplot(residuals, bins=80, ax=axes[0], color="#4C78A8", edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="#E15759", linewidth=1.5, linestyle="--", label="Zero error")
axes[0].set_title("Residual distribution")
axes[0].set_xlabel("Actual − Predicted  (log scale)")
axes[0].set_ylabel("Count")
axes[0].legend(fontsize=9)
axes[0].text(0.97, 0.97, "Should be centred at 0",
             transform=axes[0].transAxes, ha="right", va="top",
             fontsize=9, color="#666666")

# 2 — Residuals vs. predicted (hexbin instead of overplotted scatter)
hb = axes[1].hexbin(ols_pred, residuals, gridsize=60, cmap="Blues", mincnt=1, linewidths=0.1)
axes[1].axhline(0, color="#E15759", linewidth=1.5, linestyle="--")
axes[1].set_title("Residuals vs. predicted")
axes[1].set_xlabel("Predicted  log(trip_duration)")
axes[1].set_ylabel("Residual")
fig.colorbar(hb, ax=axes[1], label="Trips")
axes[1].text(0.97, 0.97, "Random scatter = good fit",
             transform=axes[1].transAxes, ha="right", va="top",
             fontsize=9, color="#666666")

# 3 — Actual vs. predicted (hexbin)
lims = [y_val.min(), y_val.max()]
hb2 = axes[2].hexbin(y_val, ols_pred, gridsize=60, cmap="Greens", mincnt=1, linewidths=0.1)
axes[2].plot(lims, lims, color="#E15759", linewidth=1.5, linestyle="--", label="Perfect prediction")
axes[2].set_title("Actual vs. predicted")
axes[2].set_xlabel("Actual  log(trip_duration)")
axes[2].set_ylabel("Predicted  log(trip_duration)")
axes[2].legend(fontsize=9)
fig.colorbar(hb2, ax=axes[2], label="Trips")
axes[2].text(0.05, 0.95, f"RMSE = {ols_rmse:.4f}",
             transform=axes[2].transAxes, ha="left", va="top",
             fontsize=10, color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

fig.suptitle("OLS regression — residual diagnostics", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ols_residuals.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: ols_residuals.png")

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

