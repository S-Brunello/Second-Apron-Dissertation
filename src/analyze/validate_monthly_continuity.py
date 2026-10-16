
"""Validate monthly continuity against the existing annual panel.

Read-only validation. Does not modify any existing datasets.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MONTHLY = ROOT / "data/processed/team_month_continuity.csv"
ANNUAL = ROOT / "data/processed/team_season_panel.csv"

monthly = pd.read_csv(MONTHLY)
annual = pd.read_csv(ANNUAL)

# Existing ESPN seasons use the ending calendar year:
# 2020 = 2019-20, 2021 = 2020-21, etc.
monthly["season"] = (
    (monthly["season"] - 1).astype(str)
    + "-"
    + monthly["season"].astype(str).str[-2:]
)


monthly = monthly.rename(columns={"team": "team_id"})

# Standardize ESPN team abbreviations to existing panel IDs
team_id_map = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS",
}

monthly["team_id"] = (
    monthly["team_id"]
    .replace(team_id_map)
)

# Aggregate using minutes, NOT a simple monthly average.
reconstructed = (
    monthly.groupby(["team_id", "season"], as_index=False)
    .agg(
        returning_minutes=("returning_minutes", "sum"),
        total_minutes=("total_minutes", "sum"),
        months=("month", "nunique"),
        games=("games", "sum")
    )
)

reconstructed["continuity_monthly_aggregated"] = (
    reconstructed["returning_minutes"]
    / reconstructed["total_minutes"]
)

# Keep precise existing annual continuity values.
annual = annual[
    ["team_id", "season", "continuity"]
].dropna(
    subset=["continuity"]
).rename(
    columns={"continuity": "continuity_existing"}
)

# Restrict the comparison to seasons with existing annual values
reconstructed = reconstructed[
    reconstructed["season"].isin(annual["season"])
].copy()

comparison = reconstructed.merge(
    annual,
    on=["team_id", "season"],
    how="left",
    validate="one_to_one"
)

if comparison["continuity_existing"].isna().any():
    missing = comparison[
        comparison["continuity_existing"].isna()
    ]
    raise ValueError(
        f"Missing annual comparisons:\n{missing.to_string(index=False)}"
    )

comparison["difference_pp"] = (
    comparison["continuity_monthly_aggregated"]
    - comparison["continuity_existing"]
) * 100

print("\nANNUAL VALIDATION")
print("Compared team-seasons:", len(comparison))

print("\nABSOLUTE DIFFERENCES (PERCENTAGE POINTS)")
absolute = comparison["difference_pp"].abs()

print("Mean:", round(absolute.mean(), 4))
print("Median:", round(absolute.median(), 4))
print("Maximum:", round(absolute.max(), 4))

print("\nTEN LARGEST DIFFERENCES")
print(
    comparison.reindex(
        absolute.sort_values(ascending=False).index
    ).head(10)[
        [
            "team_id",
            "season",
            "continuity_existing",
            "continuity_monthly_aggregated",
            "difference_pp",
            "games"
        ]
    ].to_string(index=False)
)

print("\nDIFFERENCES BY SEASON")
print(
    comparison.assign(
        abs_difference_pp=absolute
    ).groupby("season").agg(
        teams=("team_id", "nunique"),
        mean_abs_difference_pp=(
            "abs_difference_pp", "mean"
        ),
        max_abs_difference_pp=(
            "abs_difference_pp", "max"
        )
    ).round(4).to_string()
)

print("\nVALIDATION COMPLETE: Existing datasets unchanged.")
