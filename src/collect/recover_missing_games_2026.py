
from pathlib import Path
from collections import defaultdict
import re
import unicodedata

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/raw/source_tests/espn_player_box.parquet"
OUTPUT = ROOT / "data/raw/source_tests/recovered_games_2026.csv"

GAMES = {
    "401810816": ("0022500961", {"MEM", "DET"}),
    "401810822": ("0022500967", {"UTA", "POR"}),
}


def normalise_name(name):
    name = unicodedata.normalize("NFKD", str(name))
    return "".join(
        c.lower() for c in name
        if c.isascii() and c.isalnum()
    )


def parse_minutes(value):
    s = str(value).strip()

    if re.fullmatch(r"\d+:\d{2}", s):
        minutes, seconds = map(int, s.split(":"))
        return minutes + seconds / 60

    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)

    return 0.0


if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists. Will not overwrite: {OUTPUT}"
    )

# Build player-ID lookup from existing 2025-26 data.
existing = pd.read_parquet(
    SOURCE,
    columns=["season", "player_id", "name", "team"]
)

existing = existing[
    pd.to_numeric(existing["season"], errors="coerce") == 2026
].drop_duplicates()

team_lookup = defaultdict(set)
global_lookup = defaultdict(set)

for row in existing.itertuples(index=False):
    key = normalise_name(row.name)
    player_id = str(row.player_id)

    team_lookup[(str(row.team).upper(), key)].add(player_id)
    global_lookup[key].add(player_id)


def match_player(name, team):
    key = normalise_name(name)
    matches = team_lookup.get((team, key), set())

    if len(matches) != 1:
        matches = global_lookup.get(key, set())

    if len(matches) == 1:
        return next(iter(matches))

    raise ValueError(
        f"Player ID unresolved: {name}, {team}; "
        f"candidate IDs: {sorted(matches)}"
    )


records = []

for espn_id, (nba_id, expected_teams) in GAMES.items():
    url = (
        "https://site.api.espn.com/apis/site/v2/"
        f"sports/basketball/nba/summary?event={espn_id}"
    )

    response = requests.get(url, timeout=45)
    response.raise_for_status()
    data = response.json()

    groups = data.get("boxscore", {}).get("players", [])

    if not groups:
        raise ValueError(f"No player box scores: {espn_id}")

    game_records = []

    for group in groups:
        team = group["team"]["abbreviation"].upper()

        TEAM_MAP = {
            "UTAH": "UTA",
            "WSH": "WAS",
        }

        team = TEAM_MAP.get(team, team)
        for stat_group in group.get("statistics", []):
            labels = stat_group.get("labels", [])
            if "MIN" not in labels:
                raise ValueError(
                    f"MIN column missing: {espn_id}, {team}"
                )

            min_index = labels.index("MIN")

            for entry in stat_group.get("athletes", []):
                stats = entry.get("stats", [])

                if len(stats) <= min_index:
                    continue

                minutes = parse_minutes(stats[min_index])

                if minutes <= 0:
                    continue

                athlete = entry["athlete"]
                name = (
                    athlete.get("displayName")
                    or athlete.get("fullName")
                )

                player_id = match_player(name, team)

                whole_minutes = int(minutes)
                seconds = round(
                    (minutes - whole_minutes) * 60
                )

                minutes_text = (
                    f"{whole_minutes}:{seconds:02d}"
                )

                game_records.append({
                    "season": 2026,
                    "game_id": nba_id,
                    "player_id": player_id,
                    "name": name,
                    "team": team,
                    "minutes_played": minutes_text,
                    "minutes_numeric": minutes,
                    "espn_game_id": espn_id,
                    "source": "ESPN summary API",
                })

    actual_teams = {r["team"] for r in game_records}

    if actual_teams != expected_teams:
        raise ValueError(
            f"Unexpected teams for {espn_id}: {actual_teams}"
        )

    records.extend(game_records)


result = pd.DataFrame(records)

if result.duplicated(["game_id", "player_id"]).any():
    raise ValueError("Duplicate player-game records found")

if result["game_id"].nunique() != 2:
    raise ValueError("Expected exactly two games")

team_minutes = result.groupby(
    ["game_id", "team"]
)["minutes_numeric"].sum()

if len(team_minutes) != 4:
    raise ValueError("Expected four team-game observations")

if not team_minutes.between(235, 245).all():
    raise ValueError(
        f"Unexpected team playing-minute totals:\n{team_minutes}"
    )

result = result.sort_values(
    ["game_id", "team", "player_id"]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("x", encoding="utf-8", newline="") as f:
    result.to_csv(f, index=False)

print("\nRECOVERY SUCCESSFUL")
print(f"File: {OUTPUT}")
print(f"Player-game records: {len(result)}")
print(f"Unique games: {result['game_id'].nunique()}")
print("\nTeam playing-minute totals:")
print(team_minutes.to_string())
