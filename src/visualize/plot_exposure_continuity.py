
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/team_season_panel.csv"
OUTPUT = ROOT / "figures/exposure_continuity_trends.png"

df = pd.read_csv(INPUT)

# Define exposure groups using fixed pre-CBA payroll exposure.
team_exposure = df[
    ["team_id", "exposure"]
].drop_duplicates()

median_exposure = team_exposure["exposure"].median()

df["exposure_group"] = df["exposure"].apply(
    lambda x: "High exposure"
    if x > median_exposure else "Low exposure"
)

# Calculate average continuity by group and season.
trends = (
    df.dropna(subset=["continuity"])
    .groupby(["season", "exposure_group"])["continuity"]
    .mean()
    .unstack()
    .sort_index()
)

# Convert continuity into percentages.
trends = trends * 100

fig, ax = plt.subplots(figsize=(10, 6))

for group in ["High exposure", "Low exposure"]:
    ax.plot(
        trends.index,
        trends[group],
        marker="o",
        linewidth=2,
        label=group
    )

# Mark the first post-CBA season.
ax.axvline(
    x=3.5,
    color="gray",
    linestyle="--",
    alpha=0.6
) if False else None

ax.set_title(
    "NBA Roster Continuity by Pre-CBA Payroll Exposure"
)
ax.set_xlabel("Season")
ax.set_ylabel("Average Roster Continuity (%)")
ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.tick_params(axis="x", rotation=30)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fig.tight_layout()
fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

print("\nMedian exposure:", round(median_exposure, 4))
print("\nCONTINUITY TRENDS (%)")
print(trends.round(2).to_string())
print("\nSaved graph to:", OUTPUT)

plt.show()
