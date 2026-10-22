
"""Construct player-team-season salary classifications.

Preserves missing and ambiguous salaries.
Does not modify the original salary or ESPN datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SALARIES = ROOT / "data/raw/historical_player_salaries.parquet"
ESPN = ROOT / "data/raw/source_tests/espn_player_box.parquet"
OUTPUT = ROOT / "data/processed/player_salary_classification.csv"

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Review before overwriting."
    )

# NBA salary caps, indexed by season-ending year
# Sources: official NBA salary-cap announcements
salary_caps = {
    2019: 101869000,
    2020: 109140000,
    2021: 109140000,
    2022: 112414000,
    2023: 123655000,
    2024: 136021000,
    2025: 140588000,
    2026: 154647000,
}

# Standard ESPN abbreviation -> NBA team ID
team_ids = {
    "ATL": 1610612737, "BOS": 1610612738,
    "BKN": 1610612751, "BRK": 1610612751,
    "CHA": 1610612766, "CHI": 1610612741,
    "CLE": 1610612739, "DAL": 1610612742,
    "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745,
    "IND": 1610612754, "LAC": 1610612746,
    "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749,
    "MIN": 1610612750, "NOP": 1610612740,
    "NOR": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753,
    "PHI": 1610612755, "PHO": 1610612756,
    "PHX": 1610612756, "POR": 1610612757,
    "SAC": 1610612758, "SAN": 1610612759,
    "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}

salary = pd.read_parquet(SALARIES)
espn = pd.read_parquet(ESPN)

salary["player_id"] = salary["player_id"].astype("string")
espn["player_id"] = espn["player_id"].astype("string")

salary["nba_team_id"] = pd.to_numeric(
    salary["team_id"], errors="coerce"
)

# Regular-season player-team-season observations
espn = espn[
    espn["game_id"].astype(str).str.startswith("002")
].copy()

espn["nba_team_id"] = espn["team"].map(team_ids)

if espn["nba_team_id"].isna().any():
    raise ValueError("Unmapped ESPN team code")

played = espn[
    ["season", "player_id", "name", "team", "nba_team_id"]
].drop_duplicates(
    ["season", "player_id", "nba_team_id"]
).copy()

# Salary records with an actual reported amount
valid = salary.dropna(
    subset=["player_id", "salary"]
).copy()

# Exact player-team-season matches
exact = (
    valid.groupby(["season", "player_id", "nba_team_id"])
    .agg(
        exact_values=("salary", "nunique"),
        exact_salary=("salary", "first")
    )
    .reset_index()
)

# Keep exact salary only when unambiguous
exact.loc[
    exact["exact_values"] != 1, "exact_salary"
] = np.nan

# Player-season fallback, used only with one distinct salary
fallback = (
    valid.groupby(["season", "player_id"])
    .agg(
        season_values=("salary", "nunique"),
        season_salary=("salary", "first")
    )
    .reset_index()
)

fallback.loc[
    fallback["season_values"] != 1, "season_salary"
] = np.nan

df = played.merge(
    exact,
    on=["season", "player_id", "nba_team_id"],
    how="left",
    validate="one_to_one"
)

df = df.merge(
    fallback,
    on=["season", "player_id"],
    how="left",
    validate="many_to_one"
)

# Prefer exact matching; never substitute arbitrary values
df["resolved_salary"] = df["exact_salary"].fillna(
    df["season_salary"]
)

df["salary_source"] = "unresolved"

df.loc[
    df["season_values"].fillna(0) > 1,
    "salary_source"
] = "ambiguous_player_season"

df.loc[
    df["season_salary"].notna(),
    "salary_source"
] = "player_season_fallback"

df.loc[
    df["exact_salary"].notna(),
    "salary_source"
] = "exact_team_match"

# Define relative salaries
df["salary_cap"] = df["season"].map(salary_caps)

if df["salary_cap"].isna().any():
    raise ValueError("Missing NBA salary-cap information")

df["salary_cap_share"] = (
    df["resolved_salary"] / df["salary_cap"]
)

# Nullable classifications: unknown remains unknown
df["low_salary_5pct"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)
df["low_salary_10pct"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)

known = df["resolved_salary"].notna()

df.loc[known, "low_salary_5pct"] = (
    df.loc[known, "salary_cap_share"] <= 0.05
).astype(int).to_numpy()

df.loc[known, "low_salary_10pct"] = (
    df.loc[known, "salary_cap_share"] <= 0.10
).astype(int).to_numpy()

# Validate
if df.duplicated(
    ["season", "player_id", "nba_team_id"]
).any():
    raise ValueError("Duplicate player-team-season rows")

if len(df) != len(played):
    raise ValueError("Player-team-season observations changed")

if df["resolved_salary"].lt(0).any():
    raise ValueError("Negative reported salary")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nSALARY CLASSIFICATION SUMMARY")
print("Player-team-season observations:", len(df))
print("\nMATCHING STATUS")
print(df["salary_source"].value_counts().to_string())

print("\nCLASSIFICATION COVERAGE BY SEASON")
summary = df.groupby("season").agg(
    observations=("player_id", "size"),
    known_salary=("resolved_salary", "count"),
    low_5pct=("low_salary_5pct", "sum"),
    low_10pct=("low_salary_10pct", "sum")
)
summary["coverage_pct"] = (
    100 * summary["known_salary"] / summary["observations"]
).round(2)

print(summary.to_string())
print("\nSaved:", OUTPUT)
