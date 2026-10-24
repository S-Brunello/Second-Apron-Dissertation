
"""Evaluate ambiguous salary classifications across seasons.

Read-only. Does not select an arbitrary salary value.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

classification = pd.read_csv(
    ROOT / "data/processed/player_salary_classification.csv",
    dtype={"player_id": "string"}
)

raw = pd.read_parquet(
    ROOT / "data/raw/historical_player_salaries.parquet"
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

for df in [classification, raw, espn]:
    df["player_id"] = df["player_id"].astype("string")

raw["salary"] = pd.to_numeric(
    raw["salary"], errors="coerce"
)

ranges = (
    raw.groupby(["season", "player_id"])
    .agg(
        salary_count=("salary", "count"),
        min_salary=("salary", "min"),
        max_salary=("salary", "max")
    )
    .reset_index()
)

# Only previously ambiguous player-team-season records.
ambiguous = classification[
    classification["salary_source"]
    == "ambiguous_player_season"
].copy()

ambiguous = ambiguous.merge(
    ranges,
    on=["season", "player_id"],
    how="left",
    validate="many_to_one"
)

ambiguous["cap"] = ambiguous["season"].map(caps)

if ambiguous["cap"].isna().any():
    raise ValueError("Missing salary cap")

for pct in [5, 10]:
    threshold = ambiguous["cap"] * pct / 100

    ambiguous[f"status_{pct}"] = "uncertain"

    ambiguous.loc[
        ambiguous["max_salary"] <= threshold,
        f"status_{pct}"
    ] = "definitely_low"

    ambiguous.loc[
        ambiguous["min_salary"] > threshold,
        f"status_{pct}"
    ] = "definitely_high"

# Prepare playing-minute weights.
espn["game_id"] = espn["game_id"].astype("string")
espn = espn[
    espn["game_id"].str.startswith("002")
].copy()

parts = espn["minutes_played"].astype("string").str.extract(
    r"^(\d+):(\d{2})$"
).apply(pd.to_numeric, errors="coerce")

espn["minutes"] = parts[0] + parts[1] / 60
espn = espn[espn["minutes"] > 0].copy()

team_map = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS"
}

espn["team_clean"] = espn["team"].replace(team_map)
ambiguous["team_clean"] = ambiguous["team"].replace(team_map)

keys = ["season", "player_id", "team_clean"]

if ambiguous.duplicated(keys).any():
    raise ValueError("Duplicate classification keys")

games = espn.merge(
    ambiguous[
        keys + ["status_5", "status_10"]
    ],
    on=keys,
    how="inner",
    validate="many_to_one"
)

print("\nAMBIGUOUS SALARY THRESHOLD AUDIT")

for pct in [5, 10]:
    print(f"\n{pct}% SALARY-CAP THRESHOLD")

    summary = (
        games.groupby(
            ["season", f"status_{pct}"]
        )["minutes"]
        .sum()
        .unstack(fill_value=0)
    )

    for col in ["definitely_low", "definitely_high", "uncertain"]:
        if col not in summary.columns:
            summary[col] = 0.0

    summary["total"] = summary[
        ["definitely_low", "definitely_high", "uncertain"]
    ].sum(axis=1)

    summary["resolved_pct"] = (
        100 * (
            summary["definitely_low"]
            + summary["definitely_high"]
        ) / summary["total"]
    )

    print(
        summary[
            ["definitely_low", "definitely_high",
             "uncertain", "resolved_pct"]
        ].round(2).to_string()
    )

print("\nNo files changed.")
