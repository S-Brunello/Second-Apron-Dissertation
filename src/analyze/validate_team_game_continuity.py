
"""Validate game-level continuity against the existing annual panel.

Read-only: does not change any existing datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

GAME_FILE = ROOT / "data/processed/team_game_continuity.csv"
ANNUAL_FILE = ROOT / "data/processed/team_season_panel.csv"

games = pd.read_csv(GAME_FILE, dtype={"game_id": str})
annual = pd.read_csv(ANNUAL_FILE)

# 1. Check basic game-level integrity

assert not games.duplicated(["team_id", "game_id"]).any()
assert games["continuity"].between(0, 1).all()
assert games["total_minutes"].gt(0).all()

# Every unique game should have two teams.
teams_per_game = games.groupby("game_id")["team_id"].nunique()
assert teams_per_game.eq(2).all()

# 2. Reconstruct annual continuity

reconstructed = (
    games.groupby(
        ["team_id", "nba_season"], as_index=False
    )
    .agg(
        total_minutes=("total_minutes", "sum"),
        returning_minutes=("returning_minutes", "sum"),
        games=("game_id", "nunique")
    )
)

reconstructed["continuity_game_aggregated"] = (
    reconstructed["returning_minutes"]
    / reconstructed["total_minutes"]
)

reconstructed = reconstructed.rename(
    columns={"nba_season": "season"}
)

# 3. Compare with existing annual panel

annual = (
    annual[["team_id", "season", "continuity"]]
    .dropna(subset=["continuity"])
    .rename(columns={
        "continuity": "continuity_existing"
    })
)

comparison = reconstructed.merge(
    annual,
    on=["team_id", "season"],
    how="inner",
    validate="one_to_one"
)

expected = len(annual)

if len(comparison) != expected:
    raise ValueError(
        f"Matched {len(comparison)} of {expected} "
        "existing annual observations."
    )

comparison["difference_pp"] = (
    comparison["continuity_game_aggregated"]
    - comparison["continuity_existing"]
) * 100

absolute = comparison["difference_pp"].abs()

# 4. Display validation results

print("\nTEAM-GAME TO ANNUAL VALIDATION")
print("Compared team-seasons:", len(comparison))

print("\nABSOLUTE DIFFERENCES (PERCENTAGE POINTS)")
print("Mean:", round(absolute.mean(), 8))
print("Median:", round(absolute.median(), 8))
print("Maximum:", round(absolute.max(), 8))

print("\nDIFFERENCES BY SEASON")
print(
    comparison.assign(
        absolute_difference_pp=absolute
    )
    .groupby("season")
    .agg(
        teams=("team_id", "nunique"),
        mean_absolute_pp=("absolute_difference_pp", "mean"),
        max_absolute_pp=("absolute_difference_pp", "max")
    )
    .round(8)
    .to_string()
)

print("\nTEN LARGEST DIFFERENCES")
print(
    comparison.reindex(
        absolute.sort_values(ascending=False).index
    ).head(10)[[
        "team_id",
        "season",
        "continuity_existing",
        "continuity_game_aggregated",
        "difference_pp",
        "games"
    ]].to_string(index=False)
)

if np.allclose(
    comparison["continuity_game_aggregated"],
    comparison["continuity_existing"],
    rtol=0,
    atol=1e-10
):
    print("\nPASS: Annual continuity reproduced.")
else:
    print("\nCHECK: Differences require investigation.")

print("\nExisting datasets unchanged.")
