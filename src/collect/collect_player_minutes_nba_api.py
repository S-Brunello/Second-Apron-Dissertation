"""Download NBA regular-season player/team MIN totals for roster continuity.

#Research definition (implemented by build_roster_continuity.py): a returning
player has positive minutes for the SAME team in consecutive seasons. Leaving
and returning within the current season does not break eligibility; returning
after missing the immediately previous season does NOT count. A traded player's
minutes are counted separately for each team; never use league-wide totals.

Data source: NBA Stats LeagueDashPlayerStats, queried for EACH team separately
with PerMode=Totals and SeasonType='Regular Season'. The explicit TeamID filter
prevents league-wide multi-team aggregates from replacing team-specific stints.

Input: data/processed/team_exposure.csv with team_id, team_name, season.
Output: data/raw/player_team_season_minutes.csv, columns:
    player_id,team_id,season,minutes

Each successful team-season response is cached as CSV in
    data/raw/nba_minutes_cache/
This allows resuming after NBA Stats timeouts/rate-limits without starting over.
No final output is written unless ALL expected 270 team-seasons pass validation.

Install: python -m pip install nba_api pandas requests
Run from repository root: python src/collect/collect_player_minutes.py
Options: --sleep 2.5 --attempts 4 --refresh
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPOSURE = ROOT / "data" / "processed" / "team_exposure.csv"
CACHE = ROOT / "data" / "raw" / "nba_minutes_cache"
OUTPUT = ROOT / "data" / "raw" / "player_team_season_minutes.csv"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2017, 2026)]
FIELDS = ["player_id", "team_id", "season", "minutes"]

# Common Basketball Reference/Spotrac abbreviations vs NBA Stats abbreviations.
# This is only a fallback; matching to official franchise names is preferred.
ALIASES = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX", "NJN": "BKN",
           "NOH": "NOP", "NOK": "NOP", "GS": "GSW", "NY": "NYK",
           "SA": "SAS", "UTAH": "UTA"}


def norm(value: object) -> str:
    return "".join(c.lower() for c in str(value) if c.isalnum())


def load_teams():
    from nba_api.stats.static import teams as nba_teams

    if not EXPOSURE.exists():
        raise FileNotFoundError(f"Missing {EXPOSURE}")
    ref = pd.read_csv(EXPOSURE, dtype="string")
    required = {"team_id", "team_name", "season"}
    if not required.issubset(ref):
        raise ValueError(f"team_exposure.csv missing {sorted(required - set(ref.columns))}")
    ref = ref.loc[ref.season == "2022-23", ["team_id", "team_name"]].copy()
    if len(ref) != 30 or ref.team_id.nunique() != 30:
        raise ValueError("Expected exactly 30 unique teams for 2022-23 in team_exposure.csv")

    official = nba_teams.get_teams()
    by_name = {norm(t["full_name"]): t for t in official}
    by_abbr = {t["abbreviation"]: t for t in official}
    mapped = []
    for r in ref.itertuples(index=False):
        local_id, team_name = str(r.team_id).strip(), str(r.team_name).strip()
        match = by_name.get(norm(team_name))
        if match is None:
            match = by_abbr.get(ALIASES.get(local_id.upper(), local_id.upper()))
        if match is None:
            raise ValueError(f"Cannot map {local_id!r} / {team_name!r} to NBA team ID")
        mapped.append((local_id, int(match["id"]), match["full_name"]))
    if len({nba_id for _, nba_id, _ in mapped}) != 30:
        raise ValueError("Multiple local IDs mapped to the same official NBA team; inspect mapping")
    print("Team identifiers (Repository -> NBA Stats):")
    for local_id, nba_id, full_name in sorted(mapped):
        print(f"  {local_id:5} -> {nba_id}  {full_name}")
    return mapped


def validate_part(frame: pd.DataFrame, team_id: str, season: str) -> pd.DataFrame:
    if not set(FIELDS).issubset(frame):
        raise ValueError(f"{team_id} {season}: missing columns {set(FIELDS) - set(frame.columns)}")
    part = frame[FIELDS].copy()
    part["player_id"] = part.player_id.astype("string").str.strip()
    part["team_id"] = part.team_id.astype("string").str.strip()
    part["season"] = part.season.astype("string").str.strip()
    part["minutes"] = pd.to_numeric(part.minutes, errors="coerce")
    if part.empty or part.player_id.isna().any() or (part.player_id == "").any():
        raise ValueError(f"{team_id} {season}: empty response or missing player IDs")
    if not (part.team_id == team_id).all() or not (part.season == season).all():
        raise ValueError(f"{team_id} {season}: response has unexpected team/season rows")
    if part.minutes.isna().any() or (part.minutes < 0).any():
        raise ValueError(f"{team_id} {season}: invalid minutes")
    if part.duplicated(["player_id", "team_id", "season"]).any():
        raise ValueError(f"{team_id} {season}: duplicate player-team-season rows")
    part = part.loc[part.minutes > 0].copy()
    # A full team season generally totals many thousands of player-minutes.
    # This is deliberately loose to accommodate lockouts/shortened seasons,
    # but catches accidentally truncated or per-game responses.
    total = float(part.minutes.sum())
    if not 12000 <= total <= 26000:
        raise ValueError(f"{team_id} {season}: suspicious team minutes total {total:.0f}; inspect source")
    return part

# Short 45 second timeout to give the NBA Stats API time to respond without letting one request block the entire dataset completely
def fetch_part(nba_id: int, local_id: str, season: str):
    from nba_api.stats.endpoints import leaguedashplayerstats

    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Base",
        team_id_nullable=str(nba_id),
        timeout=45,
    )
    raw = response.get_data_frames()[0]
    columns = {"PLAYER_ID", "TEAM_ID", "MIN"}
    if not columns.issubset(raw):
        raise ValueError(f"NBA response missing columns: {sorted(columns - set(raw.columns))}")
    raw["TEAM_ID"] = pd.to_numeric(raw["TEAM_ID"], errors="coerce")
    if not (raw["TEAM_ID"] == nba_id).all():
        raise ValueError(
            f"NBA returned an aggregate or unfiltered row for {local_id} {season}; refusing to mix team stints"
        )
    part = pd.DataFrame({
        "player_id": raw["PLAYER_ID"].astype("string"),
        "team_id": local_id,
        "season": season,
        "minutes": raw["MIN"],
    })
    return validate_part(part, local_id, season)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep", type=float, default=2.5,
                        help="Seconds between API requests (default 2.5)")
    parser.add_argument("--attempts", type=int, default=4,
                        help="Retry attempts per team-season (default 4)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore existing cached responses")
    args = parser.parse_args()
    if args.sleep < 0 or args.attempts < 1:
        parser.error("--sleep must be >= 0 and --attempts must be >= 1")

    try:
        import nba_api  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install dependencies: python -m pip install nba_api pandas requests") from exc

    mapping = load_teams()
    CACHE.mkdir(parents=True, exist_ok=True)
    pieces = []
    failures = []
    jobs = [(local, nba, season) for season in SEASONS for local, nba, _ in mapping]
    for index, (local_id, nba_id, season) in enumerate(jobs, 1):
        cache_file = CACHE / f"{season}_{local_id}.csv"
        label = f"[{index}/{len(jobs)}] {season} {local_id}"
        if cache_file.exists() and not args.refresh:
            try:
                part = validate_part(pd.read_csv(cache_file), local_id, season)
                pieces.append(part)
                print(label, "cached")
                continue
            except Exception as exc:
                print(label, f"invalid cache ({exc}); retrying download")
        for attempt in range(1, args.attempts + 1):
            try:
                part = fetch_part(nba_id, local_id, season)
                # Cache only complete, validated team-season responses.
                part.to_csv(cache_file, index=False)
                pieces.append(part)
                print(label, f"downloaded; {len(part)} players, {part.minutes.sum():.0f} minutes")
                break
            except Exception as exc:
                print(label, f"attempt {attempt}/{args.attempts} failed: {exc}")
                if attempt < args.attempts:
                    time.sleep(max(5.0, args.sleep * attempt * 2))
        else:
            failures.append((local_id, season))
        time.sleep(args.sleep)

    if failures:
        print("\nINCOMPLETE DOWNLOAD. Failed team-seasons:")
        for team, season in failures:
            print(f"  {season} {team}")
        raise SystemExit(
            "No final CSV written. Existing caches were kept: fix connection / NBA Stats access "
            "and rerun the same command to resume."
        )

    final = pd.concat(pieces, ignore_index=True)
    if final.duplicated(["player_id", "team_id", "season"]).any():
        raise ValueError("Duplicate player-team-season keys in combined dataset")
    groups = final.groupby(["team_id", "season"]).size()
    if len(groups) != 270:
        raise ValueError(f"Expected 270 team-seasons (30 x 9), found {len(groups)}")
    final = final.sort_values(["season", "team_id", "player_id"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT, index=False)
    print("\nPLAYER MINUTES COLLECTION COMPLETE")
    print("-------------------------------")
    print("Player-team-season rows:", len(final))
    print("Team-seasons:", len(groups))
    print("Seasons:", len(SEASONS))
    print("Saved:", OUTPUT)
    print("\nNEXT:")
    print("python src/construct/build_roster_continuity.py")
    print("python src/construct/build_team_season_panel.py")


if __name__ == "__main__":
    main()
