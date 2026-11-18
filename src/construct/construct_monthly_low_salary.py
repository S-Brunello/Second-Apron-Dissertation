
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"

SOURCE = (
    PROCESSED / "team_game_low_salary_8season_v1.csv"
)
OUTPUT = (
    PROCESSED / "team_month_low_salary_8season_v1.csv"
)

if OUTPUT.exists():
    raise FileExistsError(
        f"Will not overwrite existing file: {OUTPUT}"
    )

# 1. Load validated team-game data
df = pd.read_csv(
    SOURCE, dtype={"game_id": "string"}
)

df["game_date"] = pd.to_datetime(
    df["game_date"], errors="raise"
)

# Match existing continuity panel's YYYY-MM format
df["month"] = df["game_date"].dt.strftime("%Y-%m")

KEYS = ["season", "team", "month"]

SUM_COLUMNS = [
    "total_minutes",
    "known_minutes_5",
    "known_minutes_10",
    "low_minutes_5",
    "low_minutes_10",
    "unknown_minutes_5",
    "unknown_minutes_10",
]

# 2. Aggregate playing minutes to team-month
aggregation = {
    col: (col, "sum") for col in SUM_COLUMNS
}

aggregation["games"] = ("game_id", "nunique")

# Keep missing historical two-way data missing,
# rather than incorrectly interpreting it as zero.
aggregation["two_way_minutes"] = (
    "two_way_minutes",
    lambda s: s.sum(min_count=1)
)

monthly = (
    df.groupby(KEYS, as_index=False)
    .agg(**aggregation)
    .sort_values(KEYS)
    .reset_index(drop=True)
)

# 3. Recalculate monthly percentages
for pct in [5, 10]:
    total = monthly["total_minutes"]
    low = monthly[f"low_minutes_{pct}"]
    known = monthly[f"known_minutes_{pct}"]
    unknown = monthly[f"unknown_minutes_{pct}"]

    monthly[f"coverage_{pct}_pct"] = (
        100 * known / total
    )

    monthly[f"low_share_{pct}_pct"] = (
        100 * low / total
    )

    monthly[f"low_share_{pct}_upper_pct"] = (
        100 * (low + unknown) / total
    )

monthly["two_way_share_pct"] = (
    100 * monthly["two_way_minutes"]
    / monthly["total_minutes"]
)

# 4. Validate
if monthly.duplicated(KEYS).any():
    raise ValueError("Duplicate team-month observations")

if not monthly["total_minutes"].gt(0).all():
    raise ValueError("Invalid team-month playing minutes")

if len(monthly) != 1620:
    raise ValueError(
        f"Unexpected monthly row count: {len(monthly)}"
    )

if not monthly.groupby(
    "season"
)["team"].nunique().eq(30).all():
    raise ValueError("Expected 30 teams per season")

for col in SUM_COLUMNS:
    if not np.isclose(
        df[col].sum(),
        monthly[col].sum(),
        atol=0.05
    ):
        raise ValueError(
            f"Aggregated minutes differ: {col}"
        )

for pct in [5, 10]:
    if not np.allclose(
        monthly[f"known_minutes_{pct}"]
        + monthly[f"unknown_minutes_{pct}"],
        monthly["total_minutes"],
        atol=0.05
    ):
        raise ValueError(
            f"Minutes do not reconcile at {pct}%"
        )

    if not monthly[
        f"low_share_{pct}_pct"
    ].between(0, 100.001).all():
        raise ValueError(
            f"Invalid low-salary share at {pct}%"
        )

    if (
        monthly[f"low_share_{pct}_pct"]
        > monthly[f"low_share_{pct}_upper_pct"]
        + 0.001
    ).any():
        raise ValueError(
            f"Lower bound exceeds upper bound: {pct}%"
        )

if (
    monthly["low_minutes_5"]
    > monthly["low_minutes_10"] + 0.05
).any():
    raise ValueError("5% minutes exceed 10% minutes")

# 5. Check compatibility with continuity dataset
continuity_path = (
    PROCESSED / "team_month_continuity.csv"
)


if continuity_path.exists():
    continuity = pd.read_csv(continuity_path)

    # Standardise historical ESPN team abbreviations
    # without modifying the original continuity dataset.
    TEAM_MAP = {
        "BRK": "BKN",
        "NOR": "NOP",
        "PHO": "PHX",
        "SAN": "SAS",
    }

    continuity["team"] = continuity["team"].replace(TEAM_MAP)

    comparison = continuity[KEYS].merge(
        monthly[KEYS],
        on=KEYS,
        how="left",
        indicator=True,
        validate="one_to_one"
    )

    missing = comparison["_merge"].eq(
        "left_only"
    ).sum()

    print(
        "\nContinuity team-months without salary match:",
        missing
    )

    if missing:
        raise ValueError(
            "Continuity and salary panel keys do not align"
        )

# 6. Summary
print("\nMONTHLY LOW-SALARY PANEL SUMMARY")
print("Total team-month observations:", len(monthly))
print("Teams:", monthly["team"].nunique())

print("\nOBSERVATIONS BY SEASON")
print(
    monthly.groupby("season")
    .agg(
        team_months=("month", "size"),
        months=("month", "nunique"),
        games=("games", "sum")
    ).to_string()
)

print("\nLEAGUE-WIDE MONTHLY DATA COVERAGE")
for season, group in monthly.groupby("season"):
    total = group["total_minutes"].sum()
    low5 = 100 * group["low_minutes_5"].sum() / total
    low10 = 100 * group["low_minutes_10"].sum() / total

    print(
        f"{season-1}-{str(season)[-2:]} | "
        f"low 5%: {low5:.2f}% | "
        f"low 10%: {low10:.2f}%"
    )

# 7. Save new dataset
monthly.to_csv(OUTPUT, index=False)

print("\nMONTHLY PANEL CREATED SUCCESSFULLY")
print("Saved:", OUTPUT)
print("Original datasets preserved.")
