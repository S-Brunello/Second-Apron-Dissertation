
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
RAW = ROOT / "data/raw"

HISTORICAL = (
    PROCESSED /
    "historical_team_game_low_salary_non2025_v2.csv"
)
SEASON_2025 = (
    PROCESSED / "team_game_low_salary_2025_v2.csv"
)
DATES = RAW / "game_dates_complete_2026_v1.csv"

OUTPUT = (
    PROCESSED / "team_game_low_salary_8season_v1.csv"
)

if OUTPUT.exists():
    raise FileExistsError(
        f"Will not overwrite existing file: {OUTPUT}"
    )

historical = pd.read_csv(
    HISTORICAL, dtype={"game_id": "string"}
)
season_2025 = pd.read_csv(
    SEASON_2025, dtype={"game_id": "string"}
)
dates = pd.read_csv(
    DATES, dtype={"game_id": "string"}
)

# Ensure the correct sources are being combined.
if set(historical["season"]) != {
    2019, 2020, 2021, 2022, 2023, 2024, 2026
}:
    raise ValueError("Unexpected historical seasons")

if len(season_2025) != 2460:
    raise ValueError("Unexpected 2024-25 row count")

if dates["game_id"].duplicated().any():
    raise ValueError("Duplicate game IDs in date lookup")

# Standardise 2024-25 to the historical naming system.
season_2025 = season_2025.rename(
    columns={"team_clean": "team"}
)

season_2025 = season_2025.merge(
    dates[["game_id", "game_date", "season"]],
    on="game_id",
    how="left",
    validate="many_to_one"
)

if season_2025["game_date"].isna().any():
    raise ValueError("Missing 2024-25 game dates")

if not season_2025["season"].eq(2025).all():
    raise ValueError("Incorrect 2024-25 season assignment")

# The 2024-25 dataset has a common known-minutes
# measure for both salary thresholds.
for threshold in (5, 10):
    season_2025[f"known_minutes_{threshold}"] = (
        season_2025["known_minutes"]
    )
    season_2025[f"unknown_minutes_{threshold}"] = (
        season_2025["unknown_minutes"]
    )
    season_2025[f"coverage_{threshold}_pct"] = (
        season_2025["coverage_pct"]
    )

# Preserve two-way data where available.
# Missing historical two-way data is NOT treated as zero.
historical["two_way_minutes"] = np.nan
historical["two_way_share_pct"] = np.nan

historical["source_panel"] = "historical_v2"
season_2025["source_panel"] = "season_2025_v2"

columns = [
    "season", "game_id", "game_date", "team",
    "total_minutes",
    "known_minutes_5", "known_minutes_10",
    "low_minutes_5", "low_minutes_10",
    "unknown_minutes_5", "unknown_minutes_10",
    "coverage_5_pct", "coverage_10_pct",
    "low_share_5_pct", "low_share_10_pct",
    "low_share_5_upper_pct",
    "low_share_10_upper_pct",
    "two_way_minutes", "two_way_share_pct",
    "source_panel"
]

for label, df in [
    ("historical", historical),
    ("2024-25", season_2025)
]:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing {label} columns: {sorted(missing)}"
        )

combined = pd.concat(
    [
        historical[columns],
        season_2025[columns]
    ],
    ignore_index=True
)

combined["game_date"] = pd.to_datetime(
    combined["game_date"], errors="raise"
)

combined = combined.sort_values(
    ["season", "game_date", "game_id", "team"]
).reset_index(drop=True)

# Validate the panel.
expected_counts = {
    2019: 2460,
    2020: 1942,
    2021: 2160,
    2022: 2460,
    2023: 2460,
    2024: 2460,
    2025: 2460,
    2026: 2460
}

counts = combined.groupby("season").size().to_dict()

if counts != expected_counts:
    raise ValueError(
        f"Unexpected season counts: {counts}"
    )

if combined.duplicated(
    ["season", "game_id", "team"]
).any():
    raise ValueError("Duplicate team-game observations")

if not combined.groupby(
    ["season", "game_id"]
).size().eq(2).all():
    raise ValueError("Every game must have two teams")

if not combined.groupby(
    "season"
)["team"].nunique().eq(30).all():
    raise ValueError("Expected 30 teams per season")

for threshold in (5, 10):
    total = combined["total_minutes"]
    known = combined[f"known_minutes_{threshold}"]
    unknown = combined[f"unknown_minutes_{threshold}"]
    low = combined[f"low_minutes_{threshold}"]

    if not np.allclose(
        known + unknown, total, atol=0.05
    ):
        raise ValueError(
            f"Minutes do not reconcile at {threshold}%"
        )

    if (low > known + 0.05).any():
        raise ValueError(
            f"Low minutes exceed known minutes: {threshold}%"
        )

    expected_share = 100 * low / total

    if not np.allclose(
        expected_share,
        combined[f"low_share_{threshold}_pct"],
        atol=0.05
    ):
        raise ValueError(
            f"Low-salary shares inconsistent at {threshold}%"
        )

if (
    combined["low_minutes_5"] >
    combined["low_minutes_10"] + 0.05
).any():
    raise ValueError("5% minutes exceed 10% minutes")

print("\nEIGHT-SEASON PANEL VALIDATION")
print("Total team-game observations:", len(combined))
print("Unique teams:", combined["team"].nunique())
print("\nSeason summaries:")

for season, group in combined.groupby("season"):
    minutes = group["total_minutes"].sum()
    low5 = 100 * group["low_minutes_5"].sum() / minutes
    low10 = 100 * group["low_minutes_10"].sum() / minutes
    coverage = (
        100 * group["known_minutes_5"].sum() / minutes
    )

    print(
        f"{season-1}-{str(season)[-2:]} | "
        f"team-games: {len(group)} | "
        f"coverage: {coverage:.3f}% | "
        f"low 5%: {low5:.2f}% | "
        f"low 10%: {low10:.2f}%"
    )

combined.to_csv(OUTPUT, index=False)

print("\nCOMBINATION SUCCESSFUL")
print("Saved:", OUTPUT)
print("Original datasets preserved.")
