"""One-season data-source audit for the second-apron dissertation.

Tests llimllib/nba_data's 2022-23 season-level player dataset, without
modifying any existing dissertation datasets.

Research requirement: roster continuity needs stable player identifiers,
separate team-specific rows for traded players, and TOTAL regular-season
minutes played for each player with each team. This audit does not assume
that the source file meets those requirements; it prints evidence to check.
"""

from pathlib import Path
import io
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "data" / "raw" / "source_tests"
TEST_FILE = TEST_DIR / "llimllib_players_2023.parquet"
URL = "https://raw.githubusercontent.com/llimllib/nba_data/main/data/players_2023.parquet"


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print("Testing 2022-23 player-season data (season END year 2023).", flush=True)
    print("Source:", URL, flush=True)

    if not TEST_FILE.exists():
        try:
            response = requests.get(URL, timeout=(15, 60))
            response.raise_for_status()
        except requests.RequestException as exc:
            sys.exit(f"Download failed: {type(exc).__name__}: {exc}")

        contents = response.content
        if contents[:4] != b"PAR1" or contents[-4:] != b"PAR1":
            print("Received non-Parquet content; first 150 bytes:", repr(contents[:150]))
            sys.exit("The URL returned something other than a Parquet file (possibly a Git LFS pointer).")
        TEST_FILE.write_bytes(contents)
        print(f"Downloaded {len(contents):,} bytes to {TEST_FILE}", flush=True)
    else:
        print(f"Using cached file: {TEST_FILE}", flush=True)

    try:
        df = pd.read_parquet(TEST_FILE)
    except Exception as exc:
        sys.exit(f"Could not read Parquet: {type(exc).__name__}: {exc}")

    print(f"\nRows: {len(df):,}; Columns: {len(df.columns)}")
    print("\nALL COLUMN NAMES AND TYPES:")
    for name, dtype in df.dtypes.items():
        print(f"  {name}: {dtype}")

    print("\nFIRST THREE RECORDS (transposed):")
    if len(df):
        print(df.head(3).T.to_string(max_rows=100))

    def candidates(terms):
        return [c for c in df.columns if any(term in c.lower() for term in terms)]

    print("\nPOSSIBLE PLAYER IDENTIFIERS:", candidates(["player_id", "person_id", "playerid"]))
    print("POSSIBLE TEAM IDENTIFIERS:", candidates(["team_id", "team_abbreviation", "team_name", "teamid"]))
    print("POSSIBLE MINUTES COLUMNS:", candidates(["min", "minute"]))
    print("POSSIBLE GAMES COLUMNS:", candidates(["gp", "games_played", "games"]))

    player_cols = candidates(["player_id", "person_id", "playerid"])
    team_cols = candidates(["team_id", "team_abbreviation", "teamid"])
    if player_cols and team_cols:
        player_col, team_col = player_cols[0], team_cols[0]
        counts = df.groupby(player_col, dropna=False)[team_col].nunique(dropna=True)
        print(f"\nDistinct players: {counts.size:,}")
        print(f"Players associated with >1 team identifier: {(counts > 1).sum():,}")
        print("EXAMPLES OF MULTI-TEAM PLAYERS (up to 12 rows):")
        multi = counts[counts > 1].index[:5]
        if len(multi):
            selected = [c for c in df.columns if c in [player_col, team_col] or c in candidates(["player_name", "min", "gp"])]
            print(df[df[player_col].isin(multi)][selected].head(12).to_string(index=False))
        else:
            print("None found: season totals may be aggregated or structured differently.")
        duplicates = df.duplicated([player_col, team_col]).sum()
        print(f"Duplicate player/team pairs: {duplicates:,}")
    else:
        print("\nCould not automatically identify BOTH player and team IDs; inspect column list.")

    print("\nIMPORTANT: Do not run the continuity pipeline yet.")
    print("Confirm whether minutes are TOTAL minutes (not minutes per game),")
    print("and whether traded players have separate team-specific records.")
    print("Test finished: existing processed files were NOT changed.")


if __name__ == "__main__":
    main()
