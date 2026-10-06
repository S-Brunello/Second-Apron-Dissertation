import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Locate the repository from this script's location
project_root = Path(__file__).resolve().parents[2]

# Load the existing exposure dataset
input_file = project_root / "data/processed/team_exposure.csv"

df = pd.read_csv(input_file)

#inspect data
print("First five rows:")
print(df.head())

print(f"\nNumber of rows: {len(df)}")

print("\nColumn names:")
print(df.columns.tolist())

#check each team appears once and exposure is valid
assert df["team_id"].is_unique, "Duplicate team ids found."
assert df["exposure"].notna().all(), "Missing exposure values found"
assert (df["exposure"]>0).all(), "Non-positive exposure values found"

#Summarize variation across teams
print("\nExposure distribution:")
print(df["exposure"].describe().round(3))

# Rank teams from highest to lowest exposure
ranked = df.sort_values(
    "exposure",
    ascending=False
).copy()

# Equal exposure values receive the same rank
ranked["exposure_rank"] = ranked["exposure"].rank(
    method="min",
    ascending=False
).astype(int)

print("\nTeams ranked by exposure:")
print(
    ranked[
        ["exposure_rank", "team_id", "payroll_millions", "exposure"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

#Display teams from lowest to highest exposure
plot_data = ranked.sort_values("exposure")

fig, ax = plt.subplots(figsize=(9,10))

ax.barh(
    plot_data["team_id"],
    plot_data["exposure"],
    color="steelblue"
)

ax.axvline(
    1,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Salary cap"
)

ax.set_xlabel("2022–23 cap hit / salary cap")
ax.set_ylabel("Team")
ax.set_title("Variation in baseline payroll exposure")
ax.legend()

fig.tight_layout()

# Save the figure
output_folder = project_root / "output/figures"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "exposure_distribution_2022_23.png"
fig.savefig(output_file, dpi=300)
plt.close(fig)

print(f"\nFigure saved to: {output_file}")

# Create a folder for audit tables
tables_folder = project_root / "output/tables"
tables_folder.mkdir(parents=True, exist_ok=True)

# Save the team ranking
ranked.to_csv(
    tables_folder / "exposure_ranking_2022_23.csv",
    index=False
)

# Save the distribution statistics
summary = df["exposure"].describe()
summary.index.name = "statistic"

summary.to_csv(
    tables_folder / "exposure_summary_2022_23.csv",
    header=["value"]
)

print("\nExposure ranking and summary saved.")