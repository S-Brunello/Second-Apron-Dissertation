import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

panel = pd.read_csv(
    project_root / "data/processed/team_panel.csv"
)

# Assign groups once, using fixed baseline exposure
baseline = panel[
    ["team_id", "exposure_2022_23"]
].drop_duplicates()

assert len(baseline) == 30
assert baseline["team_id"].is_unique

baseline = baseline.sort_values(
    ["exposure_2022_23", "team_id"]
).reset_index(drop=True)

baseline["exposure_group"] = (
    ["Lower exposure"] * 10
    + ["Middle exposure"] * 10
    + ["Higher exposure"] * 10
)

panel = panel.merge(
    baseline[["team_id", "exposure_group"]],
    on="team_id",
    validate="many_to_one"
)

# Equal-weighted average across teams in each group
trends = (
    panel.groupby(["season", "exposure_group"])["continuity"]
    .mean()
    .unstack("exposure_group")
    .sort_index()
)

groups = [
    "Lower exposure",
    "Middle exposure",
    "Higher exposure"
]

fig, ax = plt.subplots(figsize=(10, 6))

for group in groups:
    ax.plot(
        trends.index,
        trends[group],
        marker="o",
        label=group
    )

ax.set_title("Roster continuity by fixed 2022–23 payroll exposure")
ax.set_xlabel("Season")
ax.set_ylabel("Mean roster continuity")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.3)
ax.legend()

fig.tight_layout()

figures_folder = project_root / "output/figures"
tables_folder = project_root / "output/tables"

figures_folder.mkdir(parents=True, exist_ok=True)
tables_folder.mkdir(parents=True, exist_ok=True)

fig.savefig(
    figures_folder / "continuity_by_exposure_group.png",
    dpi=300
)
plt.close(fig)

trends.to_csv(
    tables_folder / "continuity_by_exposure_group.csv"
)

baseline.to_csv(
    tables_folder / "exposure_group_membership.csv",
    index=False
)

print("\nMean continuity by season and exposure group:")
print((trends[groups] * 100).round(1))

print("\nSaved figure, group means, and group membership.")

# Inspect individual team histories within each exposure group
team_histories = panel.pivot(
    index=["exposure_group", "team_id"],
    columns="season",
    values="continuity"
).sort_index()

print("\nIndividual team continuity (%):")
print((team_histories * 100).round(1).to_string())

team_histories.to_csv(
    tables_folder / "continuity_team_histories.csv"
)
print("\nHigher-exposure team histories (%):")
print(
    (team_histories.loc["Higher exposure"] * 100)
    .round(1)
    .to_string()
)

# Compare continuity between 2022–23 and 2023–24
comparison = panel.pivot(
    index="team_id",
    columns="season",
    values="continuity"
)

comparison["change_pp"] = (
    comparison["2023-24"] - comparison["2022-23"]
) * 100

comparison = comparison[
    ["2022-23", "2023-24", "change_pp"]
].join(
    baseline.set_index("team_id")[
        ["exposure_2022_23", "exposure_group"]
    ]
)

comparison = comparison.sort_values("change_pp")

print("\nContinuity changes: 2022–23 to 2023–24")
print(comparison.round(3).to_string())

comparison.to_csv(
    tables_folder / "continuity_changes_2022_23_to_2023_24.csv"
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    comparison["exposure_2022_23"],
    comparison["change_pp"],
    color="steelblue"
)

# Label every team
for team_id, row in comparison.iterrows():
    ax.annotate(
        team_id,
        (row["exposure_2022_23"], row["change_pp"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1)

ax.set_xlabel("2022–23 cap hit / salary cap")
ax.set_ylabel("Change in continuity (percentage points)")
ax.set_title("Baseline exposure and continuity change, 2022–23 to 2023–24")
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    figures_folder / "exposure_vs_continuity_change.png",
    dpi=300
)
plt.close(fig)

print("\nSaved exposure-versus-continuity-change scatterplot.")