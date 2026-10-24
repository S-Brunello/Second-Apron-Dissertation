"""Audit salary-clasification coverage across eight NBA seasons.
Measures unresolved salary information using actual playing minutes [rather than just counting player records]
READ-ONLY: doesn't modify existing files.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ESPN = ROOT / "data/raw/source_tests/espn_player_box.parquet"
SALARY = ROOT / "data/processed/player_salary_classification.csv"

espn = pd.read_parquet(ESPN)
salary = pd.read_csv(
    SALARY,
    dtype={"player_id": "string"}
)

espn["player_id"] = espn["player_id"].astype("string")
espn["game_id"] = espn["game_id"].astype("string")

# Regular-season games only
espn = espn[
    espn["game_id"].str.startswith("002")
].copy()

# Parse minutes
parts = (
    espn["minutes_played"]
    .astype("string")
    .str.extract(r"^(\d+):(\d{2})$")
    .apply(pd.to_numeric, errors="coerce")
)

if parts[1].gt(60).any():
    raise ValueError("Unexpected seconds in minutes_played")

espn["minutes"] = parts[0] + parts[1] / 60

espn = espn[
    espn["minutes"].notna()
    & espn["minutes"].gt(0)
].copy()

team_map = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS"
}

espn["team_clean"] = espn["team"].replace(team_map)
salary["team_clean"] = salary["team"].replace(team_map)

keys = ["season", "player_id", "team_clean"]

if salary.duplicated(keys).any():
    raise ValueError("Duplicate player-team-season salary rows")

espn = espn.merge(
    salary[
        keys + ["resolved_salary", "salary_source"]
    ],
    on=keys,
    how="left",
    validate="many_to_one"
)

# Distinguish salary ambiguity from missing salary records.
espn["salary_issue"] = np.select(
    [
        espn["resolved_salary"].notna(),
        espn["salary_source"].eq(
            "ambiguous_player_season"
        ).fillna(False),
        espn["salary_source"].eq(
            "unresolved"
        ).fillna(False)
    ],
    [
        "known_salary",
        "ambiguous_salary",
        "missing_salary"
    ],
    default="unmatched_classification"
)

# 1. Summary by NBA season

summary = (
    espn.groupby(
        ["season", "salary_issue"]
    )
    .agg(
        appearances=("game_id", "size"),
        minutes=("minutes", "sum")
    )
    .reset_index()
)

wide = summary.pivot(
    index="season",
    columns="salary_issue",
    values="minutes"
).fillna(0)

for col in [
    "known_salary", "ambiguous_salary",
    "missing_salary", "unmatched_classification"
]:
    if col not in wide.columns:
        wide[col] = 0.0

wide["total_minutes"] = wide[
    [
        "known_salary",
        "ambiguous_salary",
        "missing_salary",
        "unmatched_classification"
    ]
].sum(axis=1)

wide["known_pct"] = (
    100 * wide["known_salary"] / wide["total_minutes"]
)

wide["missing_pct"] = (
    100 * wide["missing_salary"] / wide["total_minutes"]
)

wide["ambiguous_pct"] = (
    100 * wide["ambiguous_salary"] / wide["total_minutes"]
)

wide["unmatched_pct"] = (
    100 * wide["unmatched_classification"]
    / wide["total_minutes"]
)

print("\nHISTORICAL SALARY COVERAGE BY SEASON")
print(
    wide[
        [
            "known_pct",
            "missing_pct",
            "ambiguous_pct",
            "unmatched_pct"
        ]
    ].round(2).to_string()
)

# 2. Unresolved players by season

unresolved = espn[
    espn["salary_issue"] != "known_salary"
].copy()

players = (
    unresolved.groupby(
        [
            "season", "player_id", "name",
            "team_clean", "salary_issue"
        ],
        as_index=False
    )
    .agg(
        appearances=("game_id", "size"),
        minutes=("minutes", "sum")
    )
)

print("\nUNRESOLVED PLAYERS BY SEASON")

for season in sorted(players["season"].unique()):
    subset = players[players["season"] == season]

    print(f"\nSEASON ENDING {season}")
    print(
        "Unique players:",
        subset["player_id"].nunique()
    )
    print(
        "Unresolved minutes:",
        round(subset["minutes"].sum(), 2)
    )

    print("\nTop 10 by unresolved minutes")
    print(
        subset.sort_values(
            "minutes", ascending=False
        )[
            [
                "name", "team_clean", "salary_issue",
                "appearances", "minutes"
            ]
        ].head(10).round(2).to_string(index=False)
    )

print("\nAudit complete. No files changed.")
