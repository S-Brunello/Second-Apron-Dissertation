import pandas as pd 
from pathlib import Path

#SETTINGS

# Official NBA salary cap for 2022-23
SALARY_CAP_2022_23 = 123_655_000

input_file = "data/processed/team_payroll_2022_23_clean.csv"
output_file = "data/processed/team_exposure.csv"

# LOAN CLEAN PAYROLL DATA

df = pd.read_csv(input_file)

print(f"Loaded {len(df)} teams.")

# CONSTRUCT EXPOSURE

# Continuous pre-CBA payroll exposure:
# team payroll as a proportion of the 2022-23 salary cap

df["exposure"] = df["payroll"] / SALARY_CAP_2022_23

# payroll in millions for easier interpretation
df["payroll_millions"] = df["payroll"] / 1_000_000

# VALIDATION
assert len(df) == 30, \
    f"Expected 30 teams, found {len(df)}"

assert df["team_id"].is_unique, \
    "Duplicate team IDs found."

assert df["exposure"].notna().all(), \
    "Missing exposure values found."

assert (df["exposure"] > 0).all(), \
    "Invalid exposure value found."

# SORT BY EXPOSURE

df = df.sort_values(
    "exposure",
    ascending=False
).reset_index(drop=True)

# SAVE

Path("data/processed").mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False)

print("\nExposure successfully constructed.")
print("\nHighest-exposure teams:")

print(
    df[
        ["team_id", "team_name", "payroll_millions", "exposure"]
    ].head(10)
)

print(f"\nSaved successfully to {output_file}")