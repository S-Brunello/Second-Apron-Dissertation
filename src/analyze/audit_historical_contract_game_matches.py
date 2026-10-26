
"""Audit historical player-game contract matching.

Pilot: 2022-23 and 2023-24.
Read-only
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CONTRACTS = ROOT / "data/processed/historical_salaryswish_contracts_v2.csv"
OVERRIDES = [
    ROOT / "data/processed/historical_contract_date_overrides.csv",
    ROOT / "data/processed/historical_contract_team_overrides.csv",
]
ESPN = ROOT / "data/raw/source_tests/espn_player_box.parquet"
DATES = ROOT / "data/raw/game_dates.csv"
CLASSIFICATION = ROOT / "data/processed/player_salary_classification.csv"

SEASONS = [2023, 2024]

team_map = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS",
}


def parse_event_date(series):
    text = series.astype("string").str.extract(
        r"([A-Za-z]{3,9}\s+\d{1,2},?\s+20\d{2})",
        expand=False
    )
    return pd.to_datetime(text, errors="coerce")


# ---------------------------------------------
# 1. Load historical contracts
# ---------------------------------------------

contracts = pd.read_csv(
    CONTRACTS, dtype={"player_id": "string"}
)

contracts = contracts[
    contracts["season"].isin(SEASONS)
].copy()

contracts["team_clean"] = (
    contracts["signing_team"].replace(team_map)
)

contracts["start_date"] = pd.to_datetime(
    contracts["signing_date"], errors="coerce"
)

contracts["waiver_date"] = parse_event_date(
    contracts["waiver_details"]
)

contracts["conversion_date"] = parse_event_date(
    contracts["conversion_details"]
)

# An end event before signing is not a valid end date.
# Flag these contracts rather than inferring a correction.
contracts["invalid_end_date"] = (
    (
        contracts["waiver_date"].notna()
        & (contracts["waiver_date"] < contracts["start_date"])
    )
    | (
        contracts["conversion_date"].notna()
        & (contracts["conversion_date"] < contracts["start_date"])
    )
    | (
        contracts["waiver_details"].notna()
        & contracts["waiver_date"].isna()
    )
)

valid_waiver = contracts["waiver_date"].where(
    contracts["waiver_date"] >= contracts["start_date"]
)

valid_conversion = contracts["conversion_date"].where(
    contracts["conversion_date"] >= contracts["start_date"]
)

contracts["end_date"] = pd.concat(
    [valid_waiver, valid_conversion], axis=1
).min(axis=1)

# Missing end_date means no documented end.
# It does NOT prove the contract stayed active indefinitely.

contracts["contract_status"] = "other"

contracts.loc[
    contracts["contract_type"].str.contains(
        "TWO-WAY CONTRACT", case=False, na=False
    ),
    "contract_status"
] = "two_way"

contracts.loc[
    contracts["contract_type"].str.contains(
        "ROOKIE CONTRACT|ROOKIE SCALE CONTRACT|VETERAN CONTRACT",
        case=False,
        na=False
    ),
    "contract_status"
] = "standard"

# Exhibit 10 and camp invites are not evidence
# of an active regular-season playing contract.


# ---------------------------------------------
# 2. Load player-game appearances
# ---------------------------------------------

games = pd.read_parquet(ESPN)

games["player_id"] = games["player_id"].astype("string")
games["game_id"] = games["game_id"].astype("string")

games = games[
    games["season"].isin(SEASONS)
    & games["game_id"].str.startswith("002")
].copy()

parts = games["minutes_played"].astype("string").str.extract(
    r"^(\d+):(\d{2})$"
).apply(pd.to_numeric, errors="coerce")

games["minutes"] = parts[0] + parts[1] / 60

games = games[games["minutes"].gt(0)].copy()

games["team_clean"] = games["team"].replace(team_map)

dates = pd.read_csv(DATES, dtype={"game_id": "string"})
dates["game_date"] = pd.to_datetime(
    dates["game_date"], errors="coerce"
)

games = games.merge(
    dates[["game_id", "game_date"]],
    on="game_id",
    how="left",
    validate="many_to_one"
)

if games["game_date"].isna().any():
    raise ValueError("Some game dates are missing")

# Restrict audit to salary-unresolved player-team cases.
classification = pd.read_csv(
    CLASSIFICATION, dtype={"player_id": "string"}
)

classification["team_clean"] = (
    classification["team"].replace(team_map)
)

keys = ["season", "player_id", "team_clean"]

if classification.duplicated(keys).any():
    raise ValueError("Duplicate classification keys")

games = games.merge(
    classification[keys + ["salary_source"]],
    on=keys,
    how="left",
    validate="many_to_one"
)

games = games[
    games["salary_source"].eq("unresolved")
].copy()

if games.duplicated(["season", "player_id", "game_id"]).any():
    raise ValueError("Duplicate player-game appearances")


# ---------------------------------------------
# 3. Load verified date overrides
# ---------------------------------------------

overrides = pd.concat(
    [
        pd.read_csv(path,dtype={"player_id":"string"})
        for path in OVERRIDES
    ],
    ignore_index=True
)

overrides = overrides[
    overrides["season"].isin(SEASONS)
].copy()

overrides["team_clean"] = overrides["team"].replace(team_map)

overrides["start_date"] = pd.to_datetime(
    overrides["start_date"]
)

overrides["end_date"] = pd.to_datetime(
    overrides["end_date_exclusive"]
)


# ---------------------------------------------
# 4. Match contracts to game dates
# ---------------------------------------------

results = []

for game in games.itertuples(index=False):

    candidates = contracts[
        (contracts["season"] == game.season)
        & (contracts["player_id"] == game.player_id)
        & (contracts["team_clean"] == game.team_clean)
        & (contracts["start_date"] <= game.game_date)
        & (
            contracts["end_date"].isna()
            | (contracts["end_date"] > game.game_date)
        )
    ].copy()

    # Do not trust invalid waiver or conversion dates.
    candidates = candidates[
        ~candidates["invalid_end_date"]
    ]

    applicable_override = overrides[
        (overrides["season"] == game.season)
        & (overrides["player_id"] == game.player_id)
        & (overrides["team_clean"] == game.team_clean)
        & (overrides["start_date"] <= game.game_date)
        & (overrides["end_date"] > game.game_date)
    ]

    status = "unmatched"
    evidence = "none"

    if len(applicable_override) == 1:
        status = applicable_override.iloc[0]["contract_status"]
        evidence = "verified_override"

    elif len(applicable_override) > 1:
        status = "conflict"
        evidence = "multiple_overrides"

    elif not candidates.empty:

        # Most recently signed contract is the first
        # candidate to inspect.
        latest_date = candidates["start_date"].max()

        latest = candidates[
            candidates["start_date"] == latest_date
        ]

        statuses = set(latest["contract_status"])

        if len(statuses) == 1:
            candidate_status = statuses.pop()

            if candidate_status in ("two_way", "standard"):
                status = candidate_status
                evidence = "dated_contract"
            else:
                status = "unmatched"
                evidence = "non_playing_contract"
        else:
            status = "conflict"
            evidence = "same_date_conflict"

    results.append({
        "season": game.season,
        "player_id": game.player_id,
        "name": game.name,
        "team": game.team_clean,
        "game_id": game.game_id,
        "game_date": game.game_date,
        "minutes": game.minutes,
        "status": status,
        "evidence": evidence,
    })

matches = pd.DataFrame(results)

if matches.empty:
    raise ValueError("No games matched the audit selection")


# ---------------------------------------------
# 5. Summarise matching results
# ---------------------------------------------

print("\nPLAYER-GAME CONTRACT MATCHING PILOT")

summary = matches.groupby(
    ["season", "status"]
).agg(
    appearances=("game_id", "size"),
    minutes=("minutes", "sum")
).reset_index()

print(summary.round(2).to_string(index=False))

print("\nMATCHING COVERAGE BY SEASON")

for season, subset in matches.groupby("season"):
    total = subset["minutes"].sum()

    classified = subset[
        subset["status"].isin(["two_way", "standard"])
    ]["minutes"].sum()

    print(
        f"{season - 1}-{str(season)[-2:]}: "
        f"{100 * classified / total:.2f}% "
        f"of previously unresolved minutes classified"
    )

print("\nUNMATCHED AND CONFLICTING PLAYERS")

problem = matches[
    matches["status"].isin(["unmatched", "conflict"])
]

if problem.empty:
    print("None")
else:
    unresolved = problem.groupby(
        ["season", "name", "team", "status"],
        as_index=False
    ).agg(
        appearances=("game_id", "size"),
        minutes=("minutes", "sum")
    )

    print(
        unresolved.sort_values(
            ["season", "minutes"],
            ascending=[True, False]
        ).round(2).to_string(index=False)
    )
print("\nMATCHING EVIDENCE")
print(
    matches.groupby(["season", "evidence"])
    .agg(
        appearances=("game_id", "size"),
        minutes=("minutes", "sum")
    )
    .round(2)
    .to_string()
)
print("\nNo files changed.")
