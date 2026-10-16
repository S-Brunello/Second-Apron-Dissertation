
"""Visualize pre-CBA payroll exposure across NBA teams.

Uses the existing 2022-23 exposure dataset.
Does not modify source data or existing regressions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/processed/team_exposure.csv"
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load existing exposure data
df = pd.read_csv(SOURCE)

assert len(df) == 30, "Expected 30 NBA teams."
assert df["team_id"].nunique() == 30
assert df["exposure"].notna().all()

exposure = df["exposure"]

# -----------------------------
# FIGURE 1: EXPOSURE HISTOGRAM
# -----------------------------

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.hist(
    exposure,
    bins=8,
    edgecolor="black",
    alpha=0.8
)

ax.axvline(
    exposure.median(),
    color="darkred",
    linestyle="--",
    linewidth=1.8,
    label=f"Median = {exposure.median():.3f}"
)

ax.axvline(
    exposure.mean(),
    color="darkblue",
    linestyle=":",
    linewidth=1.8,
    label=f"Mean = {exposure.mean():.3f}"
)

ax.set_title(
    "Distribution of Pre-CBA Payroll Exposure",
    fontsize=14
)

ax.set_xlabel(
    "2022-23 Team Payroll / NBA Salary Cap"
)

ax.set_ylabel("Number of Teams")
ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT / "exposure_histogram.png",
    dpi=300
)

plt.close()

# -----------------------------
# FIGURE 2: TEAM EXPOSURE PLOT
# -----------------------------

ranked = df.sort_values("exposure")

fig, ax = plt.subplots(figsize=(9, 9))

ax.scatter(
    ranked["exposure"],
    ranked["team_name"],
    s=55
)

ax.axvline(
    exposure.median(),
    color="darkred",
    linestyle="--",
    label="Median exposure"
)

ax.set_title(
    "Pre-CBA Payroll Exposure by NBA Team",
    fontsize=14
)

ax.set_xlabel(
    "2022-23 Team Payroll / NBA Salary Cap"
)

ax.set_ylabel("")
ax.grid(axis="x", alpha=0.2)
ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT / "exposure_team_dotplot.png",
    dpi=300
)

plt.close()

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------

print("\nEXPOSURE DISTRIBUTION")
print(exposure.describe().round(4))

print("\nSKEWNESS")
print(round(exposure.skew(), 4))

print("\nTEAM RANKINGS")
print(
    df.sort_values(
        "exposure", ascending=False
    )[["team_id", "team_name", "exposure"]]
    .to_string(index=False)
)

print("\nFIGURES SAVED:")
print(OUTPUT / "exposure_histogram.png")
print(OUTPUT / "exposure_team_dotplot.png")
