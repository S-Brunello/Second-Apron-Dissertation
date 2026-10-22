import pandas as pd

ROOT = "data/"

# 1. LOAD DATA

contracts = pd.read_csv(
    ROOT + "processed/player_contracts_2025_clean.csv",
    dtype={"player_id": "string"}
)

overrides = pd.read_csv(
    ROOT + "processed/contract_interval_overrides_2025.csv",
    dtype={"player_id": "string"}
)

espn = pd.read_parquet(
    ROOT + "raw/source_tests/espn_player_box.parquet"
)

dates = pd.read_csv(
    ROOT + "raw/game_dates.csv",
    dtype={"game_id": "string"}
)

espn["player_id"] = espn["player_id"].astype("string")
espn["game_id"] = espn["game_id"].astype("string")

# 2. PREPARE PLAYER-GAME DATA

games = espn[
    (espn["season"] == 2025)
    & espn["game_id"].str.startswith("002")
    & espn["player_id"].isin(contracts["player_id"])
].copy()

games = games.merge(
    dates[["game_id", "game_date"]],
    on="game_id",
    how="left",
    validate="many_to_one"
)

if games["game_date"].isna().any():
    raise ValueError("Some games have no date")

parts = (
    games["minutes_played"]
    .astype("string")
    .str.extract(r"^(\d+):(\d{2})$")
    .apply(pd.to_numeric, errors="coerce")
)

games["minutes"] = parts[0] + parts[1] / 60

games = games[
    games["minutes"].notna()
    & games["minutes"].gt(0)
].copy()

games["game_date"] = pd.to_datetime(
    games["game_date"]
)

games["team_clean"] = games["team"].replace({
    "BRK": "BKN",
    "NOR": "NOP",
    "PHO": "PHX",
    "SAN": "SAS"
})

keys = ["player_id", "game_id", "team_clean"]

if games.duplicated(keys).any():
    raise ValueError(
        "Duplicate player-team-game observations"
    )

# 3. PREPARE ORIGINAL CONTRACT INTERVALS

contracts["start"] = pd.to_datetime(
    contracts["signing_date"]
)

contracts["end"] = pd.to_datetime(
    contracts["waiver_date"]
)

def classify_contract(contract_type):
    text = str(contract_type).upper()

    if text.startswith("TWO-WAY"):
        return "two_way"

    if "ROOKIE" in text or "VETERAN" in text:
        return "standard"

    return "preseason_or_other"

contracts["status"] = contracts[
    "contract_type"
].apply(classify_contract)

# Flag invalid extracted dates.
invalid_end = (
    contracts["end"].notna()
    & (contracts["end"] <= contracts["start"])
)

print("\nINVALID ORIGINAL CONTRACT END DATES")
print(
    contracts.loc[
        invalid_end,
        ["name", "contract_type",
         "signing_team", "start", "end"]
    ].to_string(index=False)
    if invalid_end.any() else "None"
)

# Invalid end dates are treated as unknown.
# The overrides supply corrected intervals where available.
contracts.loc[invalid_end, "end"] = pd.NaT

# Exhibit 10 alone is not evidence of regular-season status.
contracts = contracts[
    contracts["status"] != "preseason_or_other"
].copy()

# 4. MATCH ORIGINAL CONTRACTS

original = games.merge(
    contracts[
        [
            "player_id",
            "signing_team",
            "start",
            "end",
            "status"
        ]
    ],
    on="player_id",
    how="left"
)

original["valid"] = (
    original["team_clean"].eq(
        original["signing_team"]
    )
    & original["game_date"].ge(
        original["start"]
    )
    & (
        original["end"].isna()
        | original["game_date"].lt(
            original["end"]
        )
    )
)

original = original[
    original["valid"]
].copy()

original["priority"] = 1

# 5. MATCH CORRECTED CONTRACT INTERVALS

overrides["start"] = pd.to_datetime(
    overrides["start_date"]
)

overrides["end"] = pd.to_datetime(
    overrides["end_date_exclusive"]
)

# Rename the contract's team before merging [fixes previous error].
override_columns = overrides[
    [
        "player_id",
        "team",
        "start",
        "end",
        "contract_status"
    ]
].rename(
    columns={
        "team": "contract_team",
        "contract_status": "status"
    }
)

corrected = games.merge(
    override_columns,
    on="player_id",
    how="left"
)

corrected["valid"] = (
    corrected["team_clean"].eq(
        corrected["contract_team"]
    )
    & corrected["game_date"].ge(
        corrected["start"]
    )
    & (
        corrected["end"].isna()
        | corrected["game_date"].lt(
            corrected["end"]
        )
    )
)

corrected = corrected[
    corrected["valid"]
].copy()

corrected["priority"] = 0

# 6. COMBINE MATCHES

matches = pd.concat(
    [
        corrected[
            keys + ["status", "priority", "start"]
        ],
        original[
            keys + ["status", "priority", "start"]
        ]
    ],
    ignore_index=True
)

if not matches.empty:

    # give verified overrides priority.
    best_priority = matches.groupby(
        keys
    )["priority"].transform("min")

    matches = matches[
        matches["priority"] == best_priority
    ].copy()

    # Where multiple applicable contracts remain, choose the most recently signed.
    latest_start = matches.groupby(
        keys
    )["start"].transform("max")

    matches = matches[
        matches["start"] == latest_start
    ].copy()

    # Do not silently select conflicting contracts signed on the same date.
    resolved = (
        matches.groupby(
            keys, as_index=False
        )
        .agg(
            status_count=("status", "nunique"),
            status=("status", "first"),
            priority=("priority", "first")
        )
    )

    resolved.loc[
        resolved["status_count"] > 1,
        "status"
    ] = "conflict"

else:
    resolved = pd.DataFrame(
        columns=keys + ["status", "priority"]
    )

# 7. PRODUCE FINAL APPEARANCE CLASSIFICATION

result = games.merge(
    resolved[
        keys + ["status", "priority"]
    ],
    on=keys,
    how="left",
    validate="one_to_one"
)

result["status"] = result[
    "status"
].fillna("unresolved")

result["evidence"] = result[
    "priority"
].map({
    0: "override",
    1: "original_contract"
}).fillna("none")

# 8. PRINT DIAGNOSTICS

print("\n" + "=" * 55)
print("2024-25 UPDATED CONTRACT-GAME AUDIT")
print("=" * 55)

print("\nTOTAL APPEARANCES")
print(len(result))

print("\nSTATUS COUNTS")
print(
    result["status"]
    .value_counts()
    .to_string()
)

print("\nEVIDENCE SOURCES")
print(
    result["evidence"]
    .value_counts()
    .to_string()
)

print("\nUNRESOLVED OR CONFLICTING APPEARANCES")

remaining = result[
    result["status"].isin([
        "unresolved",
        "conflict"
    ])
]

if remaining.empty:
    print("None")
else:
    summary = (
        remaining.groupby(
            ["name", "team_clean", "status"],
            as_index=False
        )
        .agg(
            appearances=("game_id", "size"),
            minutes=("minutes", "sum"),
            first_game=("game_date", "min"),
            last_game=("game_date", "max")
        )
    )

    print(
        summary.sort_values(
            "appearances",
            ascending=False
        ).to_string(index=False)
    )

print("\nMINUTES BY CONTRACT STATUS")
print(
    result.groupby("status")["minutes"]
    .sum()
    .round(2)
    .to_string()
)

print("\nCLASSIFICATION COVERAGE")

classified = result[
    ~result["status"].isin(
        ["unresolved", "conflict"]
    )
]

print(
    "Classified appearances:",
    len(classified),
    "/", len(result)
)

print(
    "Appearance coverage:",
    round(100 * len(classified) / len(result), 2),
    "%"
)

print(
    "Playing-minute coverage:",
    round(
        100 * classified["minutes"].sum()
        / result["minutes"].sum(),
        2
    ),
    "%"
)

print("\nOriginal source files unchanged.")

# 9. SAVE PROVISIONAL CLASSIFICATIONS

from pathlib import Path

output = Path(
    "data/processed/"
    "player_game_contract_status_2025.csv"
)

if output.exists():
    raise FileExistsError(
        f"Output already exists: {output}"
    )

if result["status"].isin(
    ["unresolved", "conflict"]
).any():
    raise ValueError(
        "Cannot save unresolved or conflicting classifications"
    )

columns = [
    "season",
    "player_id",
    "name",
    "team_clean",
    "game_id",
    "game_date",
    "minutes",
    "status",
    "evidence",
]

result[columns].to_csv(output, index=False)

print("\nPROVISIONAL CONTRACT DATASET SAVED")
print("File:", output)
print("Player-game observations:", len(result))
print("Players:", result["player_id"].nunique())