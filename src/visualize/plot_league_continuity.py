
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/team_season_panel.csv"
OUTPUT = ROOT / "figures/league_average_continuity.png"

df = pd.read_csv(DATA)

# Calculate average continuity by season.
averages = (
    df.groupby("season")["continuity"]
    .mean()
    .dropna()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    averages.index,
    averages.values * 100,
    marker="o",
    linewidth=2
)

ax.axvline(
    x=3.5,
    color="gray",
    linestyle="--",
    alpha=0.7
) if False else None

ax.set_title("Average NBA Roster Continuity by Season")
ax.set_xlabel("NBA Season")
ax.set_ylabel("Returning Players' Share of Team Minutes (%)")
ax.set_ylim(50, 75)
ax.grid(axis="y", alpha=0.25)
ax.tick_params(axis="x", rotation=30)

fig.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")

print(f"Saved graph to: {OUTPUT}")
plt.show()
