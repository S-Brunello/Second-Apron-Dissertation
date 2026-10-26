
"""Construct team-game low-salary shares, excluding 2024-25.

Uses salary records, threshold-safe ambiguous salaries,
documented salary overrides and contract-game matches.

Uncertain playing minutes remain unclassified.
Original datasets are never modified.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"

OUTPUT = PROCESSED / "historical_team_game_low_salary_non2025_v1.csv"

if OUTPUT.exists():
    raise FileExistsError(f"Already exists: {OUTPUT}")

CAPS = {
    2019: 101869000,
    2020: 109140000,
    2021: 109140000,
    2022: 112414000,
    2023: 123655000,
    2024: 136021000,
    2026: 154647000,
}

TEAM_MAP = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS",
}

KEYS = ["season", "player_id", "team_clean"]

# 1. ESPN playing minutes

games = pd.read_parquet(
    RAW / "source_tests/espn_player_box.parquet"
)

games["player_id"] = games["player_id"].astype("string")
games["game_id"] = games["game_id"].astype("string")

games = games[
    games["season"].isin(CAPS)
    & games["game_id"].str.startswith("002")
].copy()

parts = (
    games["minutes_played"]
    .astype("string")
    .str.extract(r"^(\d+):(\d{2})$")
    .apply(pd.to_numeric, errors="coerce")
)

if parts[1].gt(60).any():
    raise ValueError("Unexpected seconds in minutes_played")

games["minutes"] = parts[0] + parts[1] / 60
games = games[games["minutes"].gt(0)].copy()

games["team_clean"] = games["team"].replace(TEAM_MAP)

if games.duplicated(["season", "player_id", "game_id"]).any():
    raise ValueError("Duplicate player-game appearances")

dates = pd.read_csv(
    RAW / "game_dates.csv",
    dtype={"game_id": "string"}
)

dates["game_date"] = pd.to_datetime(dates["game_date"])

games = games.merge(
    dates[["game_id", "game_date"]],
    on="game_id",
    how="left",
    validate="many_to_one"
)

if games["game_date"].isna().any():
    raise ValueError("Missing game dates")

games["salary_cap"] = games["season"].map(CAPS)

# 2. Existing salary classifications

salary = pd.read_csv(
    PROCESSED / "player_salary_classification.csv",
    dtype={"player_id": "string"}
)

salary["team_clean"] = salary["team"].replace(TEAM_MAP)

if salary.duplicated(KEYS).any():
    raise ValueError("Duplicate salary-classification rows")

games = games.merge(
    salary[KEYS + ["resolved_salary", "salary_source"]],
    on=KEYS,
    how="left",
    validate="many_to_one"
)

games["resolved_salary"] = pd.to_numeric(
    games["resolved_salary"], errors="coerce"
)

# 3. Salary ranges for ambiguous players

raw_salary = pd.read_parquet(
    RAW / "historical_player_salaries.parquet"
)

raw_salary["player_id"] = (
    raw_salary["player_id"].astype("string")
)
raw_salary["salary"] = pd.to_numeric(
    raw_salary["salary"], errors="coerce"
)

ranges = (
    raw_salary.groupby(["season", "player_id"])
    .agg(
        min_salary=("salary", "min"),
        max_salary=("salary", "max")
    )
    .reset_index()
)

games = games.merge(
    ranges,
    on=["season", "player_id"],
    how="left",
    validate="many_to_one"
)

# 4. Manually reviewed salary overrides

manual_paths = [
    PROCESSED / "verified_salary_overrides.csv",
    PROCESSED / "verified_salary_overrides_remaining.csv",
]

manual = pd.concat(
    [pd.read_csv(p, dtype={"player_id": "string"})
     for p in manual_paths],
    ignore_index=True
)

manual["team_clean"] = manual["team"].replace(TEAM_MAP)

# first historical override file has no player IDs.
# Resolve via the existing classification dataset.
lookup = salary[
    ["season", "name", "team_clean", "player_id"]
].drop_duplicates()

needs_id = manual["player_id"].isna()

missing_ids = manual.loc[needs_id].drop(
    columns="player_id"
).merge(
    lookup,
    on=["season", "name", "team_clean"],
    how="left",
    validate="many_to_one"
)

manual = pd.concat(
    [manual.loc[~needs_id], missing_ids],
    ignore_index=True
)

if manual["player_id"].isna().any():
    raise ValueError("Unresolved manual override player IDs")

if manual.duplicated(KEYS).any():
    raise ValueError("Duplicate manual salary overrides")

manual["annual_salary"] = pd.to_numeric(
    manual["annual_salary"], errors="raise"
)

games = games.merge(
    manual[KEYS + ["annual_salary"]],
    on=KEYS,
    how="left",
    validate="many_to_one"
)

# Overrides take precedence only for historically ambiguous or missing salary classifications.
manual_eligible = (
    games["salary_source"].eq(
        "ambiguous_player_season"
    ).fillna(False)
    | games["resolved_salary"].isna()
)

games["analysis_salary"] = games["resolved_salary"]

use_manual = (
    manual_eligible & games["annual_salary"].notna()
)

games.loc[use_manual, "analysis_salary"] = (
    games.loc[use_manual, "annual_salary"]
)

# 5. Historical contract-game classifications

contracts = pd.read_csv(
    PROCESSED / "historical_player_game_contract_matches_v5.csv",
    dtype={"player_id": "string", "game_id": "string"}
)

contracts = contracts[
    contracts["season"].isin(CAPS)
].copy()

if contracts.duplicated(
    ["season", "player_id", "game_id"]
).any():
    raise ValueError("Duplicate contract-game matches")

contracts = contracts.rename(
    columns={
        "status": "contract_status",
        "certainty": "contract_certainty"
    }
)

games = games.merge(
    contracts[
        ["season", "player_id", "game_id",
         "contract_status", "contract_certainty"]
    ],
    on=["season", "player_id", "game_id"],
    how="left",
    validate="one_to_one"
)

# 6. Classify each threshold independently

for pct in [5, 10]:

    limit = games["salary_cap"] * pct / 100

    flag = pd.Series(
        pd.NA,
        index=games.index,
        dtype="boolean"
    )

    # Exact or manually reviewed salaries.
    has_salary = games["analysis_salary"].notna()

    flag.loc[has_salary] = (
        games.loc[has_salary, "analysis_salary"]
        <= limit.loc[has_salary]
    ).to_numpy()

    # Ambiguous salaries where every candidate falls on the same side of this specif. threshold.
    ambiguous = (
        flag.isna()
        & games["salary_source"].eq(
            "ambiguous_player_season"
        ).fillna(False)
    )

    definitely_low = (
        ambiguous & games["max_salary"].le(limit)
    )

    definitely_high = (
        ambiguous & games["min_salary"].gt(limit)
    )

    flag.loc[definitely_low] = True
    flag.loc[definitely_high] = False

    # Two-way base salaries are below both thresholds. Standard-contract status alone does not provide enough evidence to infer the player's salary.
    two_way = (
        flag.isna()
        & games["contract_status"].eq("two_way").fillna(False)
    )

    flag.loc[two_way] = True

    games[f"known_{pct}"] = flag.notna()
    games[f"low_{pct}"] = flag.fillna(False).astype(bool)

    games[f"known_minutes_{pct}"] = np.where(
        games[f"known_{pct}"], games["minutes"], 0
    )

    games[f"low_minutes_{pct}"] = np.where(
        games[f"low_{pct}"], games["minutes"], 0
    )

# 7. Construct team-game observations

panel = (
    games.groupby(
        ["season", "game_id", "game_date", "team_clean"],
        as_index=False
    )
    .agg(
        total_minutes=("minutes", "sum"),
        known_minutes_5=("known_minutes_5", "sum"),
        known_minutes_10=("known_minutes_10", "sum"),
        low_minutes_5=("low_minutes_5", "sum"),
        low_minutes_10=("low_minutes_10", "sum"),
    )
)

panel = panel.rename(columns={"team_clean": "team"})

for pct in [5, 10]:
    panel[f"unknown_minutes_{pct}"] = (
        panel["total_minutes"]
        - panel[f"known_minutes_{pct}"]
    )

    panel[f"coverage_{pct}_pct"] = (
        100 * panel[f"known_minutes_{pct}"]
        / panel["total_minutes"]
    )

    panel[f"low_share_{pct}_pct"] = (
        100 * panel[f"low_minutes_{pct}"]
        / panel["total_minutes"]
    )

    panel[f"low_share_{pct}_upper_pct"] = (
        100 * (
            panel[f"low_minutes_{pct}"]
            + panel[f"unknown_minutes_{pct}"]
        ) / panel["total_minutes"]
    )

# 8. Structural validation

if not panel["coverage_5_pct"].between(0, 100.001).all():
    raise ValueError("Invalid 5% classification coverage")

if not panel["coverage_10_pct"].between(0, 100.001).all():
    raise ValueError("Invalid 10% classification coverage")

if (panel["low_minutes_5"] > panel["low_minutes_10"] + 1e-8).any():
    raise ValueError("5% low minutes exceed 10% low minutes")

if (panel["low_share_5_pct"] >
    panel["low_share_5_upper_pct"] + 1e-8).any():
    raise ValueError("Invalid lower/upper bound")

print("\nHISTORICAL LOW-SALARY PANEL SUMMARY")

for season, subset in panel.groupby("season"):
    total = subset["total_minutes"].sum()

    cov5 = 100 * subset["known_minutes_5"].sum() / total
    low5 = 100 * subset["low_minutes_5"].sum() / total
    low10 = 100 * subset["low_minutes_10"].sum() / total

    print(
        f"{season-1}-{str(season)[-2:]} | "
        f"team-games: {len(subset)} | "
        f"coverage 5%: {cov5:.3f}% | "
        f"low 5%: {low5:.2f}% | "
        f"low 10%: {low10:.2f}%"
    )

panel.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("2024-25 dataset preserved.")
