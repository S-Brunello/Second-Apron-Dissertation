
"""Collect historical NBA salaries

Retains original salary records, including missing values,
multiple contracts and team IDs. Does not resolve ambiguities.
"""

from pathlib import Path
import io

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/raw/historical_player_salaries.parquet"

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Review before overwriting."
    )

frames = []

for season in range(2019, 2027):
    url = (
        "https://basketball-data.billmill.org/nba/bbref/"
        f"bbref_salaries/season={season}/data.parquet"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    frame = pd.read_parquet(io.BytesIO(response.content))

    required = {
        "season", "team_id", "bbref_id",
        "player_id", "salary"
    }

    if not required.issubset(frame.columns):
        raise ValueError(
            f"Unexpected columns for season {season}"
        )

    if not frame["season"].eq(season).all():
        raise ValueError(
            f"Incorrect season labels for {season}"
        )

    frames.append(frame)
    print(f"Season {season}: {len(frame)} records")

salaries = pd.concat(frames, ignore_index=True)

# Preserve player identifiers as strings.
salaries["player_id"] = salaries["player_id"].astype("string")

# Keep missing salaries and multiple records.
# They will be handled explicitly during construction.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
salaries.to_parquet(OUTPUT, index=False)

print("\nHISTORICAL SALARY DATA COLLECTED")
print("Total records:", len(salaries))
print("Seasons:", salaries["season"].nunique())
print("Players with IDs:", salaries["player_id"].nunique())
print("Missing salary records:", salaries["salary"].isna().sum())
print("Saved:", OUTPUT)
