
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/team_game_low_salary_2025.csv"

if OUT.exists():
    raise FileExistsError(f"Output already exists: {OUT}")

espn = pd.read_parquet(
    ROOT / "data/raw/source_tests/espn_player_box.parquet"
)
salary = pd.read_csv(
    ROOT / "data/processed/player_salary_classification.csv",
    dtype={"player_id": "string"}
)
contracts = pd.read_csv(
    ROOT / "data/processed/player_game_contract_status_2025.csv",
    dtype={"player_id": "string", "game_id": "string"}
)
recovered = pd.read_csv(
    ROOT / "data/processed/player_contracts_2025_clean.csv",
    dtype={"player_id": "string"}
)

CAP = 140588000

espn["player_id"] = espn["player_id"].astype("string")
espn["game_id"] = espn["game_id"].astype("string")

players = espn[
    (espn["season"] == 2025)
    & espn["game_id"].str.startswith("002")
].copy()

parts = players["minutes_played"].astype("string").str.extract(
    r"^(\d+):(\d{2})$"
).apply(pd.to_numeric, errors="coerce")

players["minutes"] = parts[0] + parts[1] / 60
players = players[players["minutes"] > 0].copy()

team_map = {
    "BRK": "BKN", "NOR": "NOP",
    "PHO": "PHX", "SAN": "SAS"
}
players["team_clean"] = players["team"].replace(team_map)

keys = ["player_id", "game_id", "team_clean"]

if players.duplicated(keys).any():
    raise ValueError("Duplicate player-game records")

# Match the original salary classifications.
salary = salary[salary["season"] == 2025].copy()
salary["team_clean"] = salary["team"].replace(team_map)

players = players.merge(
    salary[
        ["player_id", "team_clean", "resolved_salary",
         "salary_source"]
    ],
    on=["player_id", "team_clean"],
    how="left",
    validate="many_to_one"
)

# Match the recovered game-level contract classifications.
contracts = contracts.rename(
    columns={"status": "contract_status"}
)

players = players.merge(
    contracts[
        ["player_id", "game_id", "team_clean",
         "contract_status", "evidence"]
    ],
    on=keys,
    how="left",
    validate="one_to_one"
)

# Verify the 80 recovered player classifications.
missing_original = players["resolved_salary"].isna()

# Unresolved original salaries can reflect either:
# (1) recovered two-way contracts, or
# (2) ambiguous salary records requiring further verification.
#
# Keep ambiguous cases as unknown rather than excluding
# their minutes from the denominator.

players["salary_unresolved"] = (
    players["resolved_salary"].isna()
    & players["contract_status"].isna()
)

print("\nUNRESOLVED SALARY OBSERVATIONS")
print("Appearances:", int(players["salary_unresolved"].sum()))
print(
    "Minutes:",
    round(
        players.loc[
            players["salary_unresolved"], "minutes"
        ].sum(),
        2
    )
)
if (
    players["resolved_salary"].notna()
    & players["contract_status"].notna()
).any():
    raise ValueError(
        "Unexpected overlap between salary and contract sources"
    )

# Recover standard-contract salary for Alex Reese.
reese = recovered[
    (recovered["name"] == "Alex Reese")
    & (recovered["contract_type"] == "ROOKIE CONTRACT")
    & (recovered["signing_team"] == "OKC")
]

if len(reese) != 1:
    raise ValueError("Alex Reese contract not uniquely identified")

reese_salary = float(reese["base_salary"].iloc[0])

players["analysis_salary"] = players["resolved_salary"]

standard_recovered = (
    players["contract_status"] == "rookie_standard"
)

players.loc[
    standard_recovered, "analysis_salary"
] = reese_salary

# Two-way status comes from the dated contract audit.
two_way = players["contract_status"].eq("two_way").fillna(False)

# Missing salaries are not automatically low-salary.
known = players["analysis_salary"].notna() | two_way

players["classification_known"] = known

for pct in [5, 10]:
    low = (
        (players["analysis_salary"] / CAP <= pct / 100)
        | two_way
    ).fillna(False)

    players[f"low_minutes_{pct}"] = (
        players["minutes"] * low.astype(int)
    )

players["two_way_minutes"] = (
    players["minutes"] * two_way.astype(int)
)

players["known_minutes"] = (
    players["minutes"] * known.astype(int)
)

# Construct team-game outcomes.
games = players.groupby(
    ["game_id", "team_clean"], as_index=False
).agg(
    total_minutes=("minutes", "sum"),
    known_minutes=("known_minutes", "sum"),
    two_way_minutes=("two_way_minutes", "sum"),
    low_minutes_5=("low_minutes_5", "sum"),
    low_minutes_10=("low_minutes_10", "sum")
)

games["coverage_pct"] = (
    100 * games["known_minutes"] / games["total_minutes"]
)

games["unknown_minutes"] = (
    games["total_minutes"] - games["known_minutes"]
)

for pct in [5, 10]:
    games[f"low_share_{pct}_upper_pct"] = (
        100 * (
            games[f"low_minutes_{pct}"]
            + games["unknown_minutes"]
        ) / games["total_minutes"]
    )

games["two_way_share_pct"] = (
    100 * games["two_way_minutes"] / games["total_minutes"]
)

for pct in [5, 10]:
    games[f"low_share_{pct}_pct"] = (
        100 * games[f"low_minutes_{pct}"]
        / games["total_minutes"]
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
games.to_csv(OUT, index=False)

print("\n2024-25 LOW-SALARY OUTCOME SUMMARY")
print("Team-game observations:", len(games))
print("Unique games:", games["game_id"].nunique())
print("Teams:", games["team_clean"].nunique())

totals = games[
    ["total_minutes", "known_minutes",
     "two_way_minutes", "low_minutes_5",
     "low_minutes_10"]
].sum()

print("\nMINUTES-WEIGHTED RESULTS")
print(
    "Salary classification coverage:",
    round(100 * totals["known_minutes"]
          / totals["total_minutes"], 2), "%"
)
print(
    "Two-way minutes share:",
    round(100 * totals["two_way_minutes"]
          / totals["total_minutes"], 2), "%"
)
print(
    "Low-salary 5% share:",
    round(100 * totals["low_minutes_5"]
          / totals["total_minutes"], 2), "%"
)
print(
    "Low-salary 10% share:",
    round(100 * totals["low_minutes_10"]
          / totals["total_minutes"], 2), "%"
)

print("\nSaved:", OUT)
