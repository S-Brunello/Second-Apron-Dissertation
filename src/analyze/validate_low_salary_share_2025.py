
"""Validate 2024-25 team-game low-salary outcomes.

Read-only: does not modify any datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/team_game_low_salary_2025.csv"

df = pd.read_csv(DATA)

required = [
    "game_id", "team_clean", "total_minutes",
    "known_minutes", "two_way_minutes",
    "low_minutes_5", "low_minutes_10",
    "coverage_pct", "two_way_share_pct",
    "low_share_5_pct", "low_share_10_pct",
]

missing_cols = sorted(set(required) - set(df.columns))
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# 1. Structural validation

assert len(df) == 2460, "Unexpected team-game count"
assert df["game_id"].nunique() == 1230
assert df["team_clean"].nunique() == 30

assert not df.duplicated(
    ["game_id", "team_clean"]
).any(), "Duplicate team-game rows"

assert df.groupby("game_id").size().eq(2).all(), (
    "Not every game contains exactly two teams"
)

assert df["total_minutes"].gt(0).all()

tol = 1e-6

assert (
    df["known_minutes"] <= df["total_minutes"] + tol
).all()

assert (
    df["two_way_minutes"] <= df["known_minutes"] + tol
).all()

assert (
    df["low_minutes_5"] <= df["low_minutes_10"] + tol
).all()

assert (
    df["low_minutes_10"] <= df["known_minutes"] + tol
).all()

for col in [
    "coverage_pct", "two_way_share_pct",
    "low_share_5_pct", "low_share_10_pct"
]:
    assert df[col].between(0, 100 + tol).all(), col

assert np.allclose(
    df["coverage_pct"],
    100 * df["known_minutes"] / df["total_minutes"]
)

assert np.allclose(
    df["low_share_5_pct"],
    100 * df["low_minutes_5"] / df["total_minutes"]
)

assert np.allclose(
    df["low_share_10_pct"],
    100 * df["low_minutes_10"] / df["total_minutes"]
)

print("\nSTRUCTURAL VALIDATION")
print("Team-game observations:", len(df))
print("Unique games:", df["game_id"].nunique())
print("Teams:", df["team_clean"].nunique())
print("All structural checks: PASSED")

# 2. Team-level, minutes-weighted outcomes

df["unknown_minutes"] = (
    df["total_minutes"] - df["known_minutes"]
)

team = df.groupby(
    "team_clean", as_index=False
).agg(
    games=("game_id", "size"),
    total_minutes=("total_minutes", "sum"),
    known_minutes=("known_minutes", "sum"),
    unknown_minutes=("unknown_minutes", "sum"),
    two_way_minutes=("two_way_minutes", "sum"),
    low_minutes_5=("low_minutes_5", "sum"),
    low_minutes_10=("low_minutes_10", "sum"),
)

team["coverage_pct"] = (
    100 * team["known_minutes"] / team["total_minutes"]
)

team["unknown_pct"] = (
    100 * team["unknown_minutes"] / team["total_minutes"]
)

team["two_way_pct"] = (
    100 * team["two_way_minutes"] / team["total_minutes"]
)

team["low_5_pct"] = (
    100 * team["low_minutes_5"] / team["total_minutes"]
)

team["low_10_pct"] = (
    100 * team["low_minutes_10"] / team["total_minutes"]
)

# The widest possible missing-data bounds.
team["low_5_upper_pct"] = (
    team["low_5_pct"] + team["unknown_pct"]
)

team["low_10_upper_pct"] = (
    team["low_10_pct"] + team["unknown_pct"]
)

print("\nTEAM-LEVEL LOW-SALARY UTILISATION")
print(
    team[
        ["team_clean", "games", "low_5_pct",
         "low_10_pct", "two_way_pct", "coverage_pct"]
    ]
    .sort_values("low_5_pct", ascending=False)
    .round(2)
    .to_string(index=False)
)

# 3. Missing-salary concentration

print("\nTEAMS WITH UNCLASSIFIED MINUTES")

unknown = team[team["unknown_minutes"] > tol].copy()

if unknown.empty:
    print("None")
else:
    print(
        unknown[
            ["team_clean", "unknown_minutes",
             "unknown_pct", "low_5_pct",
             "low_5_upper_pct"]
        ]
        .sort_values("unknown_minutes", ascending=False)
        .round(2)
        .to_string(index=False)
    )

print("\nMISSING-DATA BOUND WIDTHS")
print(
    "Largest team uncertainty (percentage points):",
    round(team["unknown_pct"].max(), 3)
)
print(
    "Median team uncertainty (percentage points):",
    round(team["unknown_pct"].median(), 3)
)

# 4. Descriptive distribution

print("\nTEAM DISTRIBUTION")
print(
    team[
        ["low_5_pct", "low_10_pct", "two_way_pct",
         "coverage_pct"]
    ].describe().round(2).to_string()
)

print("\nValidation finished. No files changed.")
