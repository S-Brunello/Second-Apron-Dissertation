
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

salary = pd.read_parquet(
    ROOT / "data/raw/historical_player_salaries.parquet"
)
classification = pd.read_csv(
    ROOT / "data/processed/player_salary_classification.csv",
    dtype={"player_id": "string"}
)
espn = pd.read_parquet(
    ROOT / "data/raw/source_tests/espn_player_box.parquet"
)

caps = {
    2019: 101869000,
    2020: 109140000,
    2021: 109140000,
    2022: 112414000,
    2023: 123655000,
    2024: 136021000,
    2025: 140588000,
    2026: 154647000,
}

salary["player_id"] = salary["player_id"].astype("string")
espn["player_id"] = espn["player_id"].astype("string")
espn["game_id"] = espn["game_id"].astype("string")

ranges = (
    salary.groupby(["season", "player_id"])
    .agg(
        min_salary=("salary", "min"),
        max_salary=("salary", "max")
    )
    .reset_index()
)

amb = classification[
    classification["salary_source"].eq(
        "ambiguous_player_season"
    )
].copy()

amb = amb.merge(
    ranges,
    on=["season", "player_id"],
    how="left",
    validate="many_to_one"
)

amb["cap"] = amb["season"].map(caps)

for pct in [5, 10]:
    limit = amb["cap"] * pct / 100

    amb[f"uncertain_{pct}"] = (
        amb["min_salary"].le(limit)
        & amb["max_salary"].gt(limit)
    )

team_map = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS",
}

amb["team_clean"] = amb["team"].replace(team_map)

espn = espn[
    espn["game_id"].str.startswith("002")
].copy()

espn["team_clean"] = espn["team"].replace(team_map)

parts = espn["minutes_played"].astype("string").str.extract(
    r"^(\d+):(\d{2})$"
).apply(pd.to_numeric, errors="coerce")

espn["minutes"] = parts[0] + parts[1] / 60
espn = espn[espn["minutes"].gt(0)].copy()

keys = ["season", "player_id", "team_clean"]

if amb.duplicated(keys).any():
    raise ValueError("Duplicate ambiguous classification keys")

merged = espn.merge(
    amb[
        keys + [
            "min_salary", "max_salary",
            "uncertain_5", "uncertain_10"
        ]
    ],
    on=keys,
    how="inner",
    validate="many_to_one"
)

summary = merged.groupby(
    ["season", "player_id", "name", "team_clean",
     "min_salary", "max_salary",
     "uncertain_5", "uncertain_10"],
    as_index=False
).agg(
    appearances=("game_id", "size"),
    minutes=("minutes", "sum")
)

for season in sorted(summary["season"].unique()):
    outstanding = summary[
        (summary["season"] == season)
        & summary["uncertain_5"]
    ].sort_values("minutes", ascending=False)

    if outstanding.empty:
        continue

    print(f"\nSEASON {season - 1}-{str(season)[-2:]}")
    print("Uncertain 5% players:", outstanding["player_id"].nunique())

    print(
        outstanding[
            ["name", "team_clean", "minutes",
             "min_salary", "max_salary",
             "uncertain_10"]
        ].round(2).to_string(index=False)
    )

print("\nNo files changed.")
