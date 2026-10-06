import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

# Load historical outcomes
continuity = pd.read_csv(
    project_root / "data/processed/team_continuity.csv"
)

# Load baseline exposure
exposure = pd.read_csv(
    project_root / "data/processed/team_exposure.csv"
)

# Check both datasets before merging
assert not continuity.duplicated(["team_id", "season"]).any(), \
    "Duplicate continuity observations."

assert exposure["team_id"].is_unique, \
    "Duplicate teams in exposure data."

assert exposure["season"].eq("2022-23").all(), \
    "Exposure file contains an unexpected baseline season."

assert set(continuity["team_id"]) == set(exposure["team_id"]), \
    "Team IDs do not match across datasets."

# Rename baseline variables to distinguish them from annual outcomes
baseline = exposure[
    ["team_id", "team_name", "season", "payroll", "exposure"]
].rename(columns={
    "season": "exposure_baseline_season",
    "payroll": "payroll_2022_23",
    "exposure": "exposure_2022_23"
})

# Attach the same baseline exposure to every season for each team
panel = continuity.merge(
    baseline,
    on="team_id",
    how="left",
    validate="many_to_one",
    indicator=True
)

assert panel["_merge"].eq("both").all(), \
    "Some continuity observations did not match."

panel = panel.drop(columns="_merge")

# Validate the completed panel
expected_seasons = {
    f"{year}-{str(year + 1)[-2:]}"
    for year in range(2018, 2026)
}

assert set(panel["season"]) == expected_seasons, \
    "Unexpected or missing seasons."

assert len(panel) == 240, \
    "Expected 30 teams × 8 seasons."

assert panel.groupby("season")["team_id"].nunique().eq(30).all(), \
    "Some seasons do not contain all 30 teams."

assert panel["continuity"].between(0, 1).all(), \
    "Missing or invalid continuity."

assert panel["exposure_2022_23"].notna().all(), \
    "Missing baseline exposure."

assert panel.groupby("team_id")["exposure_2022_23"].nunique().eq(1).all(), \
    "Baseline exposure changes within a team."

# Arrange and save
panel = panel.sort_values(
    ["team_id", "season"]
).reset_index(drop=True)

output_file = project_root / "data/processed/team_panel.csv"
panel.to_csv(output_file, index=False)

print("\nExample: Boston across seasons")
print(
    panel.loc[
        panel["team_id"].eq("BOS"),
        ["team_id", "season", "continuity", "exposure_2022_23"]
    ].to_string(index=False)
)

print("\nTeams per season:")
print(panel.groupby("season")["team_id"].nunique())

print(f"\nSaved {len(panel)} observations to: {output_file}")

# Compare two seasons without assigning a treatment date
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

