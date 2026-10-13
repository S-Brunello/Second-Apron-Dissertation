
from pathlib import Path
import pandas as pd

# ----------------------------------------
# 1. Set up file paths
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[2]

EXPOSURE_FILE = ROOT / "data" / "processed" / "team_exposure.csv"
OUTCOME_FILE = ROOT / "data" / "raw" / "team_season_outcomes.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "team_season_panel.csv"

# ----------------------------------------
# 2. Load our existing exposure data
# ----------------------------------------

exposure = pd.read_csv(EXPOSURE_FILE)

required = {"team_id", "team_name", "season", "exposure"}
missing = required - set(exposure.columns)

if missing:
    raise ValueError(f"Missing exposure columns: {missing}")

# All treatment exposure must come from 2022-23
exposure = exposure.loc[
    exposure["season"] == "2022-23",
    ["team_id", "team_name", "exposure"]
].copy()

assert len(exposure) == 30, "Expected 30 teams"
assert exposure["team_id"].is_unique, "Duplicate teams found"
assert exposure["exposure"].notna().all(), "Missing exposure"

# ----------------------------------------
# 3. Create the season list
# ----------------------------------------

seasons = pd.DataFrame({
    "season": [
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(2018, 2026)
    ]
})

# One row for each team in each season
panel = exposure.merge(seasons, how="cross")

# ----------------------------------------
# 4. Construct policy timing variables
# ----------------------------------------

panel["season_start"] = panel["season"].str[:4].astype(int)

# CBA change begins with the 2023-24 season
panel["post_cba"] = (
    panel["season_start"] >= 2023
).astype(int)

# Main post variable for fully phased-in apron rules
panel["post_full_apron"] = (
    panel["season_start"] >= 2024
).astype(int)

# Interaction used for our DiD regression
panel["did_exposure"] = (
    panel["exposure"] * panel["post_full_apron"]
)

# Relative year for future event-study analysis
panel["event_time"] = panel["season_start"] - 2024

# ----------------------------------------
# 5. Add outcomes if available
# ----------------------------------------

outcome_columns = ["continuity", "wins"]

if OUTCOME_FILE.exists():

    outcomes = pd.read_csv(OUTCOME_FILE)

    required_outcome = {"team_id", "season"}
    missing = required_outcome - set(outcomes.columns)

    if missing:
        raise ValueError(f"Missing outcome keys: {missing}")

    if outcomes.duplicated(["team_id", "season"]).any():
        raise ValueError("Duplicate team-season outcomes found")

    available_columns = [
        col for col in outcome_columns
        if col in outcomes.columns
    ]

    panel = panel.merge(
        outcomes[["team_id", "season"] + available_columns],
        on=["team_id", "season"],
        how="left",
        validate="one_to_one"
    )

    print("Merged available outcome data.")

else:
    print("No outcomes file yet. Creating panel framework.")

# Ensure outcomes exist as columns
for col in outcome_columns:
    if col not in panel.columns:
        panel[col] = float("nan")

# ----------------------------------------
# 6. Organize and validate
# ----------------------------------------

panel = panel[
    [
        "team_id",
        "team_name",
        "season",
        "season_start",
        "exposure",
        "continuity",
        "wins",
        "post_cba",
        "post_full_apron",
        "did_exposure",
        "event_time"
    ]
]

panel = panel.sort_values(
    ["team_id", "season_start"]
).reset_index(drop=True)

assert len(panel) == 240, "Panel should contain 240 observations"

assert not panel.duplicated(
    ["team_id", "season"]
).any(), "Duplicate team-season rows"

assert panel.groupby("team_id")["exposure"].nunique().eq(1).all(), (
    "2022-23 exposure must be fixed within each team"
)

# ----------------------------------------
# 7. Save the finished panel framework
# ----------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

panel.to_csv(OUTPUT_FILE, index=False)

print("\nTEAM-SEASON PANEL CREATED")
print("-------------------------")
print("Observations:", len(panel))
print("Teams:", panel["team_id"].nunique())
print("Seasons:", panel["season"].nunique())
print("Continuity observations:", panel["continuity"].notna().sum())
print("Wins observations:", panel["wins"].notna().sum())
print("\nSaved to:", OUTPUT_FILE)
