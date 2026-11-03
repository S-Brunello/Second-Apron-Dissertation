
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"

original_path = RAW / "source_tests/espn_player_box.parquet"
recovered_path = RAW / "source_tests/recovered_games_2026.csv"
dates_path = RAW / "game_dates.csv"

new_source_path = (
    RAW / "source_tests/espn_player_box_complete_2026_v1.parquet"
)
new_dates_path = RAW / "game_dates_complete_2026_v1.csv"

# Never overwrite an existing dataset.
for path in [new_source_path, new_dates_path]:
    if path.exists():
        raise FileExistsError(f"File already exists: {path}")

original = pd.read_parquet(original_path)
recovered = pd.read_csv(
    recovered_path,
    dtype={"game_id": "string", "player_id": "string"}
)
dates = pd.read_csv(
    dates_path,
    dtype={"game_id": "string"}
)

for df in [original, recovered]:
    df["game_id"] = df["game_id"].astype("string")
    df["player_id"] = df["player_id"].astype("string")
    df["season"] = pd.to_numeric(
        df["season"], errors="raise"
    ).astype(int)

expected = {"0022500961", "0022500967"}
actual = set(recovered["game_id"])

if actual != expected:
    raise ValueError(f"Unexpected recovery game IDs: {actual}")

if len(recovered) != 43:
    raise ValueError("Expected 43 recovered player records")

if not recovered["season"].eq(2026).all():
    raise ValueError("Recovered records have incorrect seasons")

if original["game_id"].isin(expected).any():
    raise ValueError("Recovered games already exist in original")

if dates["game_id"].isin(expected).any():
    raise ValueError(
        "Recovered game dates already exist; inspect before merging"
    )

required = [
    "season", "game_id", "player_id",
    "name", "team", "minutes_played"
]

if not set(required).issubset(original.columns):
    raise ValueError("Original source is missing required columns")

if not set(required).issubset(recovered.columns):
    raise ValueError("Recovery file is missing required columns")

# Keep the original source schema. Any additional original
# fields unavailable from ESPN summaries remain missing
# for these recovered appearances.
extra_columns = [
    col for col in original.columns
    if col not in recovered.columns
]

supplement = recovered.reindex(columns=original.columns)

combined = pd.concat(
    [original, supplement], ignore_index=True
)

if combined.duplicated(
    ["season", "game_id", "player_id"]
).any():
    raise ValueError("Duplicate player-game records")

new_dates = pd.DataFrame([
    {
        "game_id": "0022500961",
        "game_date": "2026-03-13",
        "season": 2026
    },
    {
        "game_id": "0022500967",
        "game_date": "2026-03-13",
        "season": 2026
    }
])

complete_dates = pd.concat(
    [dates, new_dates], ignore_index=True
)

if complete_dates["game_id"].duplicated().any():
    raise ValueError("Duplicate game IDs in date lookup")

season_games = combined[
    (combined["season"] == 2026)
    & combined["game_id"].str.startswith("002")
]

game_count = season_games["game_id"].nunique()
team_game_count = season_games[
    ["game_id", "team"]
].drop_duplicates().shape[0]

print("2025-26 unique games:", game_count)
print("2025-26 team-games:", team_game_count)
print("Recovered records:", len(recovered))
print("Original-only fields unavailable in recovery:", extra_columns)

if game_count != 1230 or team_game_count != 2460:
    raise ValueError("2025-26 season is not complete")

# Save only after all validations pass.
combined.to_parquet(new_source_path, index=False)
complete_dates.to_csv(new_dates_path, index=False)

print("\nINTEGRATION SUCCESSFUL")
print("New source:", new_source_path)
print("New dates:", new_dates_path)
print("Original files preserved")
