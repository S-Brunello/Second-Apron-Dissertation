"""Test game-level player statistics from llimllib/nba_data.

This is a READ-ONLY research data audit. It downloads 2022-23 player game logs,
inspects schema and team-specific records, and DOES NOT overwrite datasets.

Purpose: determine whether these logs can support roster continuity defined as
share of team-season minutes played by players who recorded positive minutes
for the same team in the immediately preceding season.

A player traded away and back within a season can still count as returning if
he played for that team the preceding season. A player absent for an entire
preceding season does not count as returning, even if he played there earlier.
"""
from pathlib import Path
import io
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'data' / 'raw' / 'source_tests' / 'llimllib_playerlog_2023.parquet'
URL = 'https://raw.githubusercontent.com/llimllib/nba_data/main/data/playerlog_2023.parquet'


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f'Using cached file: {DEST}', flush=True)
    else:
        print(f'Downloading 2022-23 player game logs:\n{URL}', flush=True)
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        # Verify Parquet can be read before retaining the downloaded copy.
        pd.read_parquet(io.BytesIO(response.content)).head(1)
        DEST.write_bytes(response.content)
        print(f'Downloaded {len(response.content):,} bytes', flush=True)

    df = pd.read_parquet(DEST)
    print(f'\nRows: {len(df):,}; columns: {len(df.columns)}')
    print('\nALL COLUMNS AND TYPES:')
    for key, dtype in df.dtypes.items():
        print(f'  {key}: {dtype}')

    candidates = ['player_id', 'player_name', 'team_id', 'team_abbreviation',
                  'team', 'game_id', 'game_date', 'min', 'minutes', 'mp',
                  'gp', 'year', 'season', 'matchup']
    available = [c for c in candidates if c in df.columns]
    print('\nFIRST FIVE RECORDS (selected columns):')
    print(df[available].head().to_string(index=False) if available else df.head().to_string(index=False))

    # Only report whether the essential source fields actually exist.
    print('\nFIELD CHECKS:')
    for key in ['player_id', 'team_id', 'team_abbreviation', 'game_id', 'min']:
        print(f'  {key}: {"PRESENT" if key in df.columns else "NOT FOUND"}')
    if 'min' in df:
        print(f'  min type: {df["min"].dtype}; example values: {df["min"].dropna().head(8).tolist()}')

    if {'player_id', 'team_id'}.issubset(df.columns):
        by_player = df.groupby('player_id')['team_id'].nunique()
        traded = by_player[by_player > 1]
        print(f'\nUnique players: {len(by_player):,}')
        print(f'Players associated with multiple teams: {len(traded):,}')
        if not traded.empty:
            example_ids = traded.index[:5]
            sample_cols = [c for c in ['player_id','player_name','team_abbreviation','team_id','game_id','min'] if c in df]
            print('\nSAMPLE OF MULTI-TEAM PLAYERS:')
            print(df.loc[df['player_id'].isin(example_ids), sample_cols].groupby('player_id', group_keys=False).head(3).to_string(index=False))
        else:
            print('WARNING: No multi-team player records found; investigate before building continuity.')

    if {'player_id', 'team_id', 'game_id'}.issubset(df.columns):
        dup = df.duplicated(['player_id','team_id','game_id']).sum()
        print(f'\nDuplicate player/team/game rows: {dup:,}')

    if {'team_id', 'game_id'}.issubset(df.columns):
        print(f'Distinct teams: {df.team_id.nunique()}')
        print(f'Distinct games: {df.game_id.nunique()}')

    print('\nTEST COMPLETE: No continuity or panel files modified.')
    print('NEXT: review field types, regular-season coverage, and traded-player rows before aggregation.')


if __name__ == '__main__':
    main()
