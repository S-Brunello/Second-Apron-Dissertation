
"""
TEST: ESPN Player Box Scores

Purpose:
Check whether the ESPN dataset contains complete
player-game records suitable for constructing
team-specific season minutes.

This script does NOT modify existing dissertation
datasets or calculate roster continuity.
"""

from pathlib import Path
import io
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]

URL = (
    "https://raw.githubusercontent.com/llimllib/"
    "nba_data/main/data/espn/player_box.parquet"
)

OUTPUT = (
    ROOT / "data/raw/source_tests/"
    "espn_player_box.parquet"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("Testing ESPN player box data...", flush=True)

try:
    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    print(f"Downloaded {len(response.content):,} bytes")

    OUTPUT.write_bytes(response.content)
    df = pd.read_parquet(io.BytesIO(response.content))

    print("\nDATASET STRUCTURE")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    print("\nFIRST FIVE RECORDS")
    print(df.head().to_string(index=False))

    print("\nPOTENTIAL IDENTIFIERS")
    for term in ["player", "team", "game", "season", "min", "sec"]:
        matches = [
            c for c in df.columns
            if term in c.lower()
        ]
        print(term, ":", matches)

    print("\nSEASON COVERAGE")
    if "season" in df.columns:
        print(df["season"].value_counts().sort_index())

    for col in ["game_id", "gameId"]:
        if col in df.columns:
            print("\nUnique games:", df[col].nunique())
            print("Sample game IDs:", df[col].head().tolist())
            break

    print("\nSaved to:", OUTPUT)

except Exception as e:
    print("\nTEST FAILED")
    print(type(e).__name__, str(e))
