#!/usr/bin/env python3
"""
Error analysis: where does the best model struggle?

Slices RMSE across three segments to find patterns in prediction error:

  1. Trip distance   — short / medium / long trips
  2. Time of day     — night / morning / midday / evening / rush hour
  3. Day type        — weekday vs. weekend

Why this matters (for presentation):
  A single global RMSE hides a lot. A model with RMSE 0.38 overall might have
  RMSE 0.25 for long trips and 0.60 for short ones. That tells you where the
  model is reliable and where it is not — which is what matters for a real
  transport planning application.

Input : data/processed/val_predictions.csv  (written by 06_model_advanced.py)
Output: reports/figures/error_by_distance.png
        reports/figures/error_by_hour_bucket.png
        reports/figures/error_by_day_type.png
        reports/figures/error_summary.png

Run from the project root:
    python3 scripts/07_error_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "val_predictions.csv"
FIGURES_DIR = ROOT / "reports" / "figures" / "06_error_analysis"
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
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(IN_PATH, parse_dates=["pickup_datetime"])
print(f"Loaded {len(df):,} validation predictions")

global_rmse = root_mean_squared_error(df["log_trip_duration"], df["predicted_log"])
print(f"Global RMSE: {global_rmse:.4f}")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def segment_rmse(df, segment_col):
    rows = []
    for name, group in df.groupby(segment_col, observed=True):
        rmse = root_mean_squared_error(group["log_trip_duration"], group["predicted_log"])
        rows.append({"segment": name, "rmse": rmse, "n_trips": len(group)})
    return pd.DataFrame(rows).sort_values("segment")


def rmse_bar(ax, seg_df, title, xlabel, color):
    bars = ax.bar(seg_df["segment"].astype(str), seg_df["rmse"],
                  color=color, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color="#333333")
    ax.axhline(global_rmse, color="#E15759", linewidth=1.2,
               linestyle="--", label=f"Global RMSE ({global_rmse:.3f})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE  (log scale)")
    ax.set_ylim(0, seg_df["rmse"].max() * 1.25)
    ax.legend(fontsize=9)

# ---------------------------------------------------------------------------
# 1. By distance bucket
# ---------------------------------------------------------------------------

df["distance_bucket"] = pd.cut(
    df["haversine_km"],
    bins=[0, 1, 3, 7, np.inf],
    labels=["Short\n(<1 km)", "Medium\n(1–3 km)", "Long\n(3–7 km)", "Very long\n(>7 km)"],
)

dist_rmse = segment_rmse(df, "distance_bucket")
print("\n--- RMSE by distance ---")
print(dist_rmse.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
rmse_bar(ax, dist_rmse, "Prediction error by trip distance", "Distance bucket", "#4C78A8")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "error_by_distance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: error_by_distance.png")

# ---------------------------------------------------------------------------
# 2. By time of day
# ---------------------------------------------------------------------------

def hour_bucket(h):
    if 0 <= h < 6:   return "Night\n(0–6 h)"
    if 6 <= h < 10:  return "Morning\n(6–10 h)"
    if 10 <= h < 16: return "Midday\n(10–16 h)"
    if 16 <= h < 20: return "Evening\n(16–20 h)"
    return "Late night\n(20–24 h)"

HOUR_ORDER = ["Night\n(0–6 h)", "Morning\n(6–10 h)", "Midday\n(10–16 h)",
              "Evening\n(16–20 h)", "Late night\n(20–24 h)"]

df["hour_bucket"] = pd.Categorical(
    df["pickup_hour"].apply(hour_bucket),
    categories=HOUR_ORDER, ordered=True,
)

hour_rmse = segment_rmse(df, "hour_bucket")
print("\n--- RMSE by time of day ---")
print(hour_rmse.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
rmse_bar(ax, hour_rmse, "Prediction error by time of day", "Time of day", "#59A14F")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "error_by_hour_bucket.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: error_by_hour_bucket.png")

# ---------------------------------------------------------------------------
# 3. By day type
# ---------------------------------------------------------------------------

df["day_type"] = df["is_weekend"].map({True: "Weekend", False: "Weekday"})

day_rmse = segment_rmse(df, "day_type")
print("\n--- RMSE by day type ---")
print(day_rmse.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
rmse_bar(ax, day_rmse, "Prediction error by day type", "Day type", "#F28E2B")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "error_by_day_type.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: error_by_day_type.png")

# ---------------------------------------------------------------------------
# 4. Summary panel — all three segments in one figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

rmse_bar(axes[0], dist_rmse, "By distance",   "Distance bucket", "#4C78A8")
rmse_bar(axes[1], hour_rmse, "By time of day", "Time of day",    "#59A14F")
rmse_bar(axes[2], day_rmse,  "By day type",    "Day type",       "#F28E2B")

fig.suptitle("Where does the model struggle?  (red line = global RMSE)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "error_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: error_summary.png")

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

worst_dist = dist_rmse.loc[dist_rmse["rmse"].idxmax(), "segment"]
worst_hour = hour_rmse.loc[hour_rmse["rmse"].idxmax(), "segment"]

print(f"""
--- Key findings ---
Global RMSE : {global_rmse:.4f}
Worst distance segment : {worst_dist.strip()}  (RMSE {dist_rmse['rmse'].max():.4f})
Worst time segment     : {worst_hour.strip()}  (RMSE {hour_rmse['rmse'].max():.4f})
Weekday RMSE : {day_rmse.loc[day_rmse['segment']=='Weekday', 'rmse'].values[0]:.4f}
Weekend RMSE : {day_rmse.loc[day_rmse['segment']=='Weekend', 'rmse'].values[0]:.4f}
""")
