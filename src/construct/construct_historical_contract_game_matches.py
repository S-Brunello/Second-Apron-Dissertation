
"""Construct historical contract-status matches for NBA player-games.

Processes seasons 2018-19 through 2025-26, excluding 2024-25,
which already has its own completed provisional classifications.

Preserves original data and previous matching outputs.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"

OUTPUT = PROCESSED / "historical_player_game_contract_matches_v2.csv"

if OUTPUT.exists():
    raise FileExistsError(f"Output already exists: {OUTPUT}")

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2026]

TEAM_MAP = {
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS",
}

KEYS = ["season", "player_id", "team_clean"]


def event_dates(series):
    extracted = series.astype("string").str.extract(
        r"([A-Za-z]{3,9}\s+\d{1,2},?\s+20\d{2})",
        expand=False,
    )
    return pd.to_datetime(extracted, errors="coerce")


# 1. CONTRACT RECORDS

contracts = pd.read_csv(
    PROCESSED / "historical_salaryswish_contracts_v2.csv",
    dtype={"player_id": "string"},
)

contracts = contracts[
    contracts["season"].isin(SEASONS)
].copy()

contracts["team_clean"] = contracts["signing_team"].replace(TEAM_MAP)

contracts["start_date"] = pd.to_datetime(
    contracts["signing_date"], errors="coerce"
)

contracts["waiver_date"] = event_dates(
    contracts["waiver_details"]
)
contracts["conversion_date"] = event_dates(
    contracts["conversion_details"]
)

contracts["invalid_end_event"] = (
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
    | contracts["start_date"].isna()
)

waiver_valid = contracts["waiver_date"].where(
    contracts["waiver_date"] >= contracts["start_date"]
)

conversion_valid = contracts["conversion_date"].where(
    contracts["conversion_date"] >= contracts["start_date"]
)

contracts["end_date"] = pd.concat(
    [waiver_valid, conversion_valid], axis=1
).min(axis=1)

contracts["contract_status"] = "non_playing_or_unknown"

contracts.loc[
    contracts["contract_type"].str.contains(
        "TWO-WAY CONTRACT", case=False, na=False
    ),
    "contract_status"
] = "two_way"

contracts.loc[
    contracts["contract_type"].str.contains(
        "ROOKIE CONTRACT|ROOKIE SCALE CONTRACT|VETERAN CONTRACT",
        case=False, na=False
    ),
    "contract_status"
] = "standard"

# Camp invites and Exhibit 10 agreements do not alone
# establish an active regular-season player contract.


# 2. VERIFIED OVERRIDES

override_paths = [
    PROCESSED / "historical_contract_date_overrides.csv",
    PROCESSED / "historical_contract_team_overrides.csv",
]

overrides = pd.concat(
    [
        pd.read_csv(path, dtype={"player_id": "string"})
        for path in override_paths
    ],
    ignore_index=True,
)

overrides = overrides[
    overrides["season"].isin(SEASONS)
].copy()

overrides["team_clean"] = overrides["team"].replace(TEAM_MAP)
overrides["start_date"] = pd.to_datetime(overrides["start_date"])
overrides["end_date"] = pd.to_datetime(
    overrides["end_date_exclusive"]
)

if (
    overrides["start_date"].isna().any()
    or overrides["end_date"].isna().any()
    or (overrides["end_date"] <= overrides["start_date"]).any()
):
    raise ValueError("Invalid override intervals")


# 3. ESPN PLAYER-GAME DATA

games = pd.read_parquet(
    RAW / "source_tests/espn_player_box.parquet"
)

games["player_id"] = games["player_id"].astype("string")
games["game_id"] = games["game_id"].astype("string")

games = games[
    games["season"].isin(SEASONS)
    & games["game_id"].str.startswith("002")
].copy()

parts = (
    games["minutes_played"]
    .astype("string")
    .str.extract(r"^(\d+):(\d{2})$")
    .apply(pd.to_numeric, errors="coerce")
)

if parts[1].gt(60).any():
    raise ValueError("Invalid seconds in minutes_played")

games["minutes"] = parts[0] + parts[1] / 60
games = games[games["minutes"].gt(0)].copy()

games["team_clean"] = games["team"].replace(TEAM_MAP)

dates = pd.read_csv(
    RAW / "game_dates.csv",
    dtype={"game_id": "string"},
)

dates["game_date"] = pd.to_datetime(
    dates["game_date"], errors="coerce"
)

games = games.merge(
    dates[["game_id", "game_date"]],
    on="game_id",
    how="left",
    validate="many_to_one",
)

if games["game_date"].isna().any():
    raise ValueError("Missing game dates")

if games.duplicated(["season", "player_id", "game_id"]).any():
    raise ValueError("Duplicate player-game records")


# 4. RESTRICT TO MISSING-SALARY APPEARANCES

classification = pd.read_csv(
    PROCESSED / "player_salary_classification.csv",
    dtype={"player_id": "string"},
)

classification["team_clean"] = (
    classification["team"].replace(TEAM_MAP)
)

if classification.duplicated(KEYS).any():
    raise ValueError("Duplicate salary-classification keys")

games = games.merge(
    classification[KEYS + ["salary_source"]],
    on=KEYS,
    how="left",
    validate="many_to_one",
)

games = games[
    games["salary_source"].eq("unresolved")
].copy()


# 5. GAME-DATE MATCHING

results = []

for game in games.itertuples(index=False):

    # Check manually documented intervals first.
    active_overrides = overrides[
        (overrides["season"] == game.season)
        & (overrides["player_id"] == game.player_id)
        & (overrides["team_clean"] == game.team_clean)
        & (overrides["start_date"] <= game.game_date)
        & (overrides["end_date"] > game.game_date)
    ]

    status = "unmatched"
    evidence = "no_matching_contract"
    certainty = "unresolved"
    matching_contracts = 0

    if not active_overrides.empty:

        distinct_statuses = active_overrides["contract_status"].unique()

        if len(distinct_statuses) == 1:
            status = distinct_statuses[0]
            evidence = "documented_override"
            certainty = "documented"
        else:
            status = "conflict"
            evidence = "conflicting_overrides"

    else:

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

        matching_contracts = len(candidates)

        if not candidates.empty:

            latest_date = candidates["start_date"].max()

            latest = candidates[
                candidates["start_date"] == latest_date
            ]

            statuses = latest["contract_status"].unique()

            if len(statuses) > 1:
                status = "conflict"
                evidence = "same_date_status_conflict"

            elif len(statuses) == 1:

                candidate = latest.iloc[0]
                candidate_status = statuses[0]

                if candidate_status in ("two_way", "standard"):

                    status = candidate_status

                    if latest["invalid_end_event"].any():
                        evidence = "invalid_end_event"
                        certainty = "provisional"

                    elif latest["end_date"].isna().any():
                        evidence = "open_end_contract"
                        certainty = "provisional"

                    else:
                        evidence = "dated_contract"
                        certainty = "dated"

                else:
                    evidence = "non_playing_contract"

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
        "certainty": certainty,
        "matching_contracts": matching_contracts,
    })


matches = pd.DataFrame(results)

if matches.empty:
    raise ValueError("No eligible player-game appearances found")

if len(matches) != len(games):
    raise ValueError("Player-game counts changed during matching")

if matches.duplicated(
    ["season", "player_id", "game_id"]
).any():
    raise ValueError("Duplicate matching results")


# 6. VALIDATION AND OUTPUT

summary = (
    matches.groupby(["season", "status"])
    .agg(
        appearances=("game_id", "size"),
        minutes=("minutes", "sum"),
    )
    .reset_index()
)

print("\nHISTORICAL CONTRACT MATCHING SUMMARY")
print(summary.round(2).to_string(index=False))

print("\nCLASSIFICATION COVERAGE BY SEASON")

for season, subset in matches.groupby("season"):

    total = subset["minutes"].sum()

    classified = subset[
        subset["status"].isin(["two_way", "standard"])
    ]["minutes"].sum()

    documented = subset[
        subset["certainty"].eq("documented")
    ]["minutes"].sum()

    print(
        f"{season - 1}-{str(season)[-2:]} | "
        f"Classified: {100 * classified / total:.2f}% | "
        f"Override-documented: {100 * documented / total:.2f}%"
    )

print("\nMATCHING EVIDENCE")
print(
    matches.groupby(["season", "evidence"])
    .agg(
        appearances=("game_id", "size"),
        minutes=("minutes", "sum"),
    )
    .round(2)
    .to_string()
)

print("\nUNMATCHED / CONFLICTING / PROVISIONAL MINUTES")

needs_review = matches[
    matches["certainty"].isin(["provisional", "unresolved"])
]

if needs_review.empty:
    print("None")
else:
    print(
        needs_review.groupby(
            ["season", "name", "team", "evidence"],
            as_index=False
        )
        .agg(
            appearances=("game_id", "size"),
            minutes=("minutes", "sum"),
        )
        .sort_values(
            ["season", "minutes"],
            ascending=[True, False],
        )
        .round(2)
        .to_string(index=False)
    )

matches.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Existing datasets preserved.")
