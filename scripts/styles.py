"""Shared colour palette and matplotlib theme for all pipeline scripts."""

import matplotlib.pyplot as plt

BG    = "#0d1117"
FG    = "#F9FAFB"
FAINT = "#D1D5DB"
GRID  = "#1E2736"
CYAN  = "#22D3EE"
AMBER = "#F59E0B"
GREEN = "#10B981"
RED   = "#F87171"


def apply_theme():
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
