
"""Construct team-game roster continuity from existing ESPN player boxes.

Definition:
Share of current-game minutes played by players who recorded
positive minutes for the same team in the preceding NBA season.

Uses regular-season games only.
Does not overwrite existing annual or monthly datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PLAYER_BOX = ROOT / "data/raw/source_tests/espn_player_box.parquet"
GAME_DATES = ROOT / "data/raw/game_dates.csv"
EXPOSURE = ROOT / "data/processed/team_exposure.csv"
OUTPUT = ROOT / "data/processed/team_game_continuity.csv"


# 1. Load existing source data

players = pd.read_parquet(PLAYER_BOX)
dates = pd.read_csv(GAME_DATES, dtype={"game_id": str})
exposure = pd.read_csv(EXPOSURE)

players = players[
    ["game_id", "player_id", "team", "season", "minutes_played"]
].copy()

for col in ["game_id", "player_id", "team"]:
    players[col] = players[col].astype(str).str.strip()

players["season"] = players["season"].astype(int)

# Regular season only
players = players[
    players["game_id"].str.startswith("002")
].copy()

# 2. Parse playing time

parts = players["minutes_played"].astype(str).str.extract(
    r"^(\d+):(\d{2})$"
)

parts = parts.apply(pd.to_numeric, errors="coerce")

# Inspect unexpected values rather than silently misclassify them
invalid = (
    parts.isna().any(axis=1)
    | parts[1].gt(60)
)

# Non-playing rows may have missing/unparseable minutes.
# They cannot contribute positive playing minutes.
if invalid.any():
    print(
        "Rows with unparseable or invalid minutes:",
        int(invalid.sum())
    )

players["minutes"] = parts[0] + parts[1] / 60

# ESPN sometimes uses MM:60; numerically this is MM+1 minutes.
players = players[
    players["minutes"].notna()
    & (players["minutes"] > 0)
].copy()

# 3. Determine previous-season team membership

membership = players[
    ["season", "team", "player_id"]
].drop_duplicates()

previous_membership = membership.copy()

# A player in season 2022 can be a returner in season 2023.
previous_membership["season"] += 1
previous_membership["returning"] = 1

players = players.merge(
    previous_membership,
    on=["season", "team", "player_id"],
    how="left",
    validate="many_to_one"
)

players["returning"] = (
    players["returning"].fillna(0).astype(int)
)

# The first available season has no preceding-season reference.
first_season = players["season"].min()

players = players[
    players["season"] > first_season
].copy()

players["returning_minutes"] = (
    players["minutes"] * players["returning"]
)

# 4. Aggregate to team-game level

games = (
    players.groupby(
        ["team", "season", "game_id"],
        as_index=False
    )
    .agg(
        total_minutes=("minutes", "sum"),
        returning_minutes=("returning_minutes", "sum"),
        players_used=("player_id", "nunique")
    )
)

games["continuity"] = (
    games["returning_minutes"] / games["total_minutes"]
)

games["continuity_pct"] = 100 * games["continuity"]

# 5. Add dates and fixed exposure

dates["game_id"] = dates["game_id"].astype(str).str.strip()

if dates["game_id"].duplicated().any():
    raise ValueError("Duplicate game IDs in game_dates.csv")

games = games.merge(
    dates[["game_id", "game_date"]],
    on="game_id",
    how="left",
    validate="many_to_one"
)

if games["game_date"].isna().any():
    raise ValueError("Missing game dates")

games["game_date"] = pd.to_datetime(games["game_date"])

# Standardize ESPN abbreviations
team_map = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS"
}

games["team_id"] = games["team"].replace(team_map)

exposure = exposure[
    ["team_id", "exposure"]
].drop_duplicates()

games = games.merge(
    exposure,
    on="team_id",
    how="left",
    validate="many_to_one"
)

if games["exposure"].isna().any():
    raise ValueError("Some teams are missing exposure values")

# 6. Construct time-period identifiers


games["season_label"] = (
    (games["season"] - 1).astype(str)
    + "-"
    + games["season"].astype(str).str[-2:]
)

# Calendar-month indicator
games["calendar_month"] = (
    games["game_date"].dt.to_period("M").astype(str)
)

# Calendar-quarter indicator
games["calendar_quarter"] = (
    games["game_date"].dt.to_period("Q").astype(str)
)

# NBA-season time indicator
games["nba_season"] = games["season_label"]

# 7. Validate output

if games.duplicated(["team_id", "game_id"]).any():
    raise ValueError("Duplicate team-game observations")

if not games["continuity"].between(0, 1).all():
    raise ValueError("Invalid continuity values")

if games["total_minutes"].le(0).any():
    raise ValueError("Non-positive team-game minutes")

games = games.sort_values(
    ["season", "game_date", "game_id", "team_id"]
).reset_index(drop=True)

columns = [
    "team_id",
    "game_id",
    "game_date",
    "season",
    "nba_season",
    "calendar_month",
    "calendar_quarter",
    "exposure",
    "total_minutes",
    "returning_minutes",
    "continuity",
    "continuity_pct",
    "players_used"
]

games = games[columns]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Review it before overwriting."
    )

games.to_csv(OUTPUT, index=False)

print("\nTEAM-GAME DATASET SUMMARY")
print("Observations:", len(games))
print("Teams:", games["team_id"].nunique())
print("NBA seasons:", games["nba_season"].nunique())
print("Unique games:", games["game_id"].nunique())

print("\nTEAM-GAME OBSERVATIONS BY SEASON")
print(games.groupby("nba_season").size().to_string())

print("\nCONTINUITY SUMMARY (%)")
print(games["continuity_pct"].describe().round(2))

print("\nFIRST FIVE ROWS")
print(games.head().to_string(index=False))

print("\nSAVED:", OUTPUT)
