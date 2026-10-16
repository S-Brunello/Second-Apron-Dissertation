
"""Construct monthly NBA roster continuity from ESPN player-game data.

Definition: share of team-month minutes played by players who
recorded positive minutes for the same team in the previous season.

Preserves existing annual datasets and regressions.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/source_tests/espn_player_box.parquet"
DATES = ROOT / "data/raw/game_dates.csv"
DEST = ROOT / "data/processed/team_month_continuity.csv"

# 1. Load existing data
df = pd.read_parquet(SOURCE)
dates = pd.read_csv(DATES, dtype={"game_id": str})

df["game_id"] = df["game_id"].astype(str)
df["season"] = df["season"].astype(int)
df["player_id"] = df["player_id"].astype(str)
df["team"] = df["team"].astype(str).str.strip()

# 2. Retain regular-season games only
df = df[df["game_id"].str.startswith("002")].copy()

# 3. Convert MM:SS into decimal minutes
parts = df["minutes_played"].str.extract(
    r"^(\d+):(\d{2})$"
)

if parts.isna().any().any():
    raise ValueError("Unexpected minutes format.")


parts = parts.astype(int)

# ESPN-derived data sometimes contain MM:60 values.
# Interpret MM:60 as (MM+1):00.
# Other seconds values above 59 remain invalid.
invalid = parts[1] > 60

if invalid.any():
    raise ValueError(
        f"Found {invalid.sum()} records with seconds > 60."
    )

rounded = (parts[1] == 60).sum()

print(
    f"Normalizing {rounded} MM:60 records "
    "to the equivalent whole-minute duration."
)

df["minutes"] = parts[0] + parts[1] / 60

assert df["minutes"].notna().all()
assert (df["minutes"] >= 0).all()

# Players with zero minutes do not count as roster members
df = df[df["minutes"] > 0].copy()

# 4. Add actual game dates
if dates["game_id"].duplicated().any():
    raise ValueError("Duplicate game IDs in date lookup.")

df = df.merge(
    dates[["game_id", "game_date"]],
    on="game_id",
    how="left",
    validate="many_to_one"
)

if df["game_date"].isna().any():
    raise ValueError("Unmatched game dates.")

df["game_date"] = pd.to_datetime(df["game_date"])
df["month"] = df["game_date"].dt.strftime("%Y-%m")

# 5. Construct season-level membership using positive minutes
membership = df[
    ["season", "team", "player_id"]
].drop_duplicates()

# Shift previous-season membership forward one season
previous = membership.copy()
previous["season"] += 1
previous["returning"] = 1

# 6. Identify returning players
df = df.merge(
    previous,
    on=["season", "team", "player_id"],
    how="left",
    validate="many_to_one"
)

df["returning"] = df["returning"].fillna(0).astype(int)

# The first season has no preceding season in our source
first_season = df["season"].min()
df = df[df["season"] > first_season].copy()

df["returning_minutes"] = (
    df["minutes"] * df["returning"]
)

# 7. Aggregate to team-month
monthly = (
    df.groupby(["season", "team", "month"], as_index=False)
      .agg(
          total_minutes=("minutes", "sum"),
          returning_minutes=("returning_minutes", "sum"),
          player_game_rows=("player_id", "size"),
          games=("game_id", "nunique")
      )
)

monthly["continuity"] = (
    monthly["returning_minutes"]
    / monthly["total_minutes"]
)

monthly["continuity_pct"] = monthly["continuity"] * 100

monthly = monthly.sort_values(
    ["season", "team", "month"]
).reset_index(drop=True)

# 8. Validation
assert monthly["continuity"].between(0, 1).all()
assert (monthly["total_minutes"] > 0).all()
assert not monthly.duplicated(
    ["season", "team", "month"]
).any()

print("\nMONTHLY PANEL SUMMARY")
print("Rows:", len(monthly))
print("Teams:", monthly["team"].nunique())
print("Seasons:", sorted(monthly["season"].unique()))
print("Months:", monthly["month"].nunique())

print("\nOBSERVATIONS BY SEASON")
print(
    monthly.groupby("season")
    .agg(
        team_months=("month", "size"),
        teams=("team", "nunique"),
        calendar_months=("month", "nunique")
    ).to_string()
)

print("\nCONTINUITY SUMMARY (%)")
print(monthly["continuity_pct"].describe().round(2))

print("\nFIRST TEN ROWS")
print(monthly.head(10).to_string(index=False))

# 9. Save as a separate dataset
if DEST.exists():
    print(f"\nNOT OVERWRITTEN: {DEST} already exists.")
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(DEST, index=False)
    print(f"\nSaved: {DEST}")
