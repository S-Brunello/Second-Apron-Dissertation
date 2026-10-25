
"""Audit remaining historical 5% salary-threshold straddlers.
Read-only. 
"""

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

target_seasons = [2019, 2020, 2022, 2026]

salary["player_id"] = salary["player_id"].astype("string")
salary["salary"] = pd.to_numeric(
    salary["salary"], errors="coerce"
)

ambiguous = classification[
    classification["season"].isin(target_seasons)
    & classification["salary_source"].eq(
        "ambiguous_player_season"
    )
].copy()

ranges = (
    salary.groupby(["season", "player_id"])
    .agg(
        min_salary=("salary", "min"),
        max_salary=("salary", "max")
    )
    .reset_index()
)

ambiguous = ambiguous.merge(
    ranges,
    on=["season", "player_id"],
    how="left",
    validate="many_to_one"
)

ambiguous["threshold_5"] = (
    ambiguous["season"].map(caps) * 0.05
)

straddlers = ambiguous[
    (ambiguous["min_salary"] <= ambiguous["threshold_5"])
    & (ambiguous["max_salary"] > ambiguous["threshold_5"])
].copy()

print("\nREMAINING HISTORICAL SALARY CASES")

for row in straddlers.sort_values(
    ["season", "name"]
).itertuples():

    print("\n" + "=" * 55)
    print(
        f"{row.name} | {row.team} | "
        f"{row.season - 1}-{str(row.season)[-2:]}"
    )
    print("=" * 55)

    print("Player ID:", row.player_id)
    print("5% threshold:", round(row.threshold_5, 2))

    records = salary[
        (salary["season"] == row.season)
        & (salary["player_id"] == row.player_id)
    ]

    print("\nAvailable salary records:")
    print(
        records[
            ["team_id", "salary"]
        ].drop_duplicates().to_string(index=False)
    )

print("\nTotal player-team cases:", len(straddlers))
print("No files changed.")
