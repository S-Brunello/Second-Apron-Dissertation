import pandas as pd
from pathlib import Path

# LOAD DATA

teams = pd.read_csv("data/raw/teams.csv")
payroll = pd.read_csv(
    "data/raw/team_payroll/team_payroll_2022_23.csv"
)

print("Teams file:", len(teams), "teams")
print("Payroll file:", len(payroll), "teams")

# CHECK DUPCLICATES
assert teams["team_id"].is_unique, \
    "Duplicate team IDs in teams.csv"

assert payroll["team_id"].is_unique, \
    "Duplicate team IDs in payroll data"

# CHECK ALL TEAMS MATCH

missing_from_payroll = set(teams["team_id"]) - set(payroll["team_id"])
unknown_in_payroll = set(payroll["team_id"]) - set(teams["team_id"])

assert not missing_from_payroll, \
    f"Teams missing from payroll data: {missing_from_payroll}"

assert not unknown_in_payroll, \
    f"Unknown teams in payroll data: {unknown_in_payroll}"

#MERGE TEAM NAMES W PAYROLL

clean = payroll.merge(
    teams,
    on="team_id",
    how="left",
    validate="one_to_one"
)

# Put columns in a sensible order
clean = clean[
    ["team_id", "team_name", "season", "payroll"]
]

# CHECKS

assert len(clean) == 30, \
    f"Expected 30 teams, found {len(clean)}"

assert clean["payroll"].notna().all(), \
    "Missing payroll values found"

assert (clean["payroll"] > 0).all(), \
    "Invalid payroll value found"

print("\nValidation successful.")
print(clean.head())

# SAVE

output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "team_payroll_2022_23_clean.csv"

clean.to_csv(output_file, index=False)

print(f"\nSaved successfully to {output_file}")