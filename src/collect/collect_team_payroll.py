import pandas as pd
from pathlib import Path

# SETTINGS

url = "https://www.salaryswish.com/past-team-payrolls/2023"
season = "2022-23"

output_folder = Path("data/raw/team_payroll")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "team_payroll_2022_23.csv"

print(f"Downloading {season} team payroll data...")

#TABLE 

tables = pd.read_html(url)
payroll = tables[0]

# Keep only the variables needed for the raw dataset
payroll = payroll[["Team", "Cap Hit"]].copy()

#EXTRACT STANDARD NBA TEAM ID

# SalarySwish includes the three-letter abbreviation at the
# end of each team name, e.g. "Boston CelticsBOS"
payroll["team_id"] = payroll["Team"].str[-3:]

#CLEAN CAP HIT

# Convert money to numbers recognizable to github
payroll["payroll"] = (
    payroll["Cap Hit"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(int)
)

# Add season
payroll["season"] = season

# FINAL RAW DATASET

payroll = payroll[
    ["team_id", "season", "payroll"]
]

# CHECKS

assert len(payroll) == 30, \
    f"Expected 30 NBA teams, found {len(payroll)}"

assert payroll["team_id"].is_unique, \
    "Duplicate team IDs found."

assert payroll["payroll"].notna().all(), \
    "Missing payroll values found."

print("\nFirst five rows:")
print(payroll.head())

print(f"\nNumber of teams: {len(payroll)}")

# SAVE
payroll.to_csv(output_file, index=False)

print(f"\nSaved successfully to {output_file}")