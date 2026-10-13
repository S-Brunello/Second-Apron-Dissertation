##Construct NBA team-season roster continuity from player-team-season minutes.

#DEFINITION (important for dissertation methodology)
#---------------------------------------------------
#A player is 'returning' for team j in season t if the player recorded
#POSITIVE minutes for that SAME team both in season t and in season t-1.

#1. If a player leaves and returns within season t, they still count as
#   returning if they played for the team in season t-1.
#2. If they played for the team in an older season but NOT in t-1,
#   they do NOT count as returning.
#3. For traded players, only minutes played for the relevant team count.
#   A league-wide aggregate row (e.g. 'TOT') is NOT a team and is excluded.
#4. This measures consecutive-season roster overlap weighted by current
#   minutes, NOT uninterrupted tenure or accumulated teammate experience.
#5. Players with zero minutes do not enter a team's observed roster.

#continuity(j,t) = current-season minutes from returning players for j
#                 ----------------------------------------------------
#                             all current-season minutes for j

#INPUT: data/raw/player_team_season_minutes.csv
#Required columns: player_id, team_id, season, minutes
#  - player_id must be a stable identifier across seasons (not just a name).
#  - team_id must use the SAME team identifiers as team_exposure.csv.
#  - season format: YYYY-YY, e.g. 2022-23.
#  - minutes must be numeric totals (not minutes per game or MM:SS).
#  - one row per player-team-season; duplicates are summed only after
#    detection and warning (e.g. repeated stints on the same team).
#  - include 2017-18 to provide the previous-season baseline for 2018-19.
#  - omit 'TOT' aggregate rows, or let this script drop them.
#
#OUTPUT: data/raw/team_season_outcomes.csv
#  Columns: team_id, season, continuity, wins (if previously present).
#  Re-run src/construct/build_team_season_panel.py to merge outcomes.
#
#This script NEVER turns missing outcomes into zero: if a team-season
#is absent from the input, it remains missing, not 0% continuity.


from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'data' / 'raw' / 'player_team_season_minutes.csv'
EXPOSURE = ROOT / 'data' / 'processed' / 'team_exposure.csv'
OUTPUT = ROOT / 'data' / 'raw' / 'team_season_outcomes.csv'
YEARS = range(2017, 2026)  # 2017-18 baseline through 2025-26
SEASONS = {f'{y}-{str(y + 1)[-2:]}' for y in YEARS}
TARGET_SEASONS = {f'{y}-{str(y + 1)[-2:]}' for y in range(2018, 2026)}


def fail(message):
    raise ValueError(message)


def main():
    if not INPUT.exists():
        sys.exit(
            f'Missing input: {INPUT}\n'
            'Create a CSV with columns: player_id,team_id,season,minutes\n'
            'Minutes must be total minutes played for that team and season.'
        )
    if not EXPOSURE.exists():
        sys.exit(f'Missing team reference: {EXPOSURE}')

    teams_df = pd.read_csv(EXPOSURE, dtype={'team_id': 'string', 'season': 'string'})
    if not {'team_id', 'season'}.issubset(teams_df.columns):
        fail('team_exposure.csv must contain team_id and season')
    team_ids = set(teams_df.loc[teams_df['season'] == '2022-23', 'team_id'].dropna())
    if len(team_ids) != 30:
        fail(f'Expected 30 team IDs from 2022-23 exposure; found {len(team_ids)}')

    df = pd.read_csv(INPUT, dtype={'player_id':'string','team_id':'string','season':'string'})
    required = {'player_id', 'team_id', 'season', 'minutes'}
    if not required.issubset(df.columns):
        fail(f'Input is missing required columns: {sorted(required - set(df.columns))}')
    df = df[list(required)].copy()
    if df[['player_id','team_id','season']].isna().any().any():
        fail('Missing player_id, team_id or season in input')
    for col in ['player_id','team_id','season']:
        df[col] = df[col].str.strip()
    if (df[['player_id','team_id','season']] == '').any().any():
        fail('Blank player_id, team_id or season in input')
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')
    if df['minutes'].isna().any() or (~df['minutes'].between(0, float('inf'))).any():
        fail('Minutes must be nonmissing, nonnegative numeric totals')

    # 'TOT' and similar league aggregate rows must NEVER inflate a team's minutes.
    aggregate = df['team_id'].isin(['TOT', '2TM', '3TM', '4TM', '5TM'])
    if aggregate.any():
        print(f'Dropped {aggregate.sum()} league-wide aggregate rows (TOT etc.)')
        df = df.loc[~aggregate].copy()

    unknown_teams = set(df['team_id']) - team_ids
    if unknown_teams:
        fail(f'Unknown team_id values: {sorted(unknown_teams)}. Harmonize IDs with team_exposure.csv.')
    invalid_seasons = set(df['season']) - SEASONS
    if invalid_seasons:
        fail(f'Unexpected season values: {sorted(invalid_seasons)}')

    duplicates = df.duplicated(['player_id','team_id','season'], keep=False)
    if duplicates.any():
        print(f'WARNING: {duplicates.sum()} rows have repeated player-team-season keys; summing their minutes.')
        print('Verify these are disjoint stints, NOT accidental duplicate downloads.')
        # Refuse automatic summing: duplicate extracts could double-count minutes.
        fail('Resolve duplicate player-team-season keys in the source CSV before continuing')

    df = df.loc[df['minutes'] > 0].copy()
    if df.empty:
        fail('No positive-minute player-team-season rows')

    # A missing prior season must not be incorrectly treated as zero returners.
    expected_source = pd.MultiIndex.from_product(
        [sorted(team_ids), sorted(SEASONS)], names=['team_id', 'season']
    )
    observed_source = pd.MultiIndex.from_frame(df[['team_id', 'season']].drop_duplicates())
    missing_source = expected_source.difference(observed_source)
    if len(missing_source):
        print('Missing source team-seasons (first 20):', list(missing_source[:20]))
        fail(f'Input incomplete: {len(missing_source)} missing team-seasons from 2017-18 through 2025-26')

    # Lookup is on same (player, team) in the IMMEDIATELY previous season.
    previous = df[['player_id','team_id','season']].copy()
    previous['season_start'] = previous['season'].str[:4].astype(int) + 1
    previous = previous[['player_id','team_id','season_start']].assign(returning=True)
    current = df.copy()
    current['season_start'] = current['season'].str[:4].astype(int)
    current = current.merge(previous, on=['player_id','team_id','season_start'],
                            how='left', validate='one_to_one')
    current['returning'] = current['returning'].fillna(False).astype(bool)
    current['returning_minutes'] = current['minutes'].where(current['returning'], 0)
    current = current.loc[current['season'].isin(TARGET_SEASONS)].copy()

    result = (current.groupby(['team_id','season'], as_index=False)
              .agg(total_minutes=('minutes','sum'),
                   returning_minutes=('returning_minutes','sum'),
                   players=('player_id','nunique'),
                   returning_players=('returning','sum')))
    result['continuity'] = result['returning_minutes'] / result['total_minutes']
    if not result['continuity'].between(0, 1).all():
        fail('Computed continuity outside [0,1]')

    # Use complete team-season grid so absent observations remain NaN.
    grid = pd.MultiIndex.from_product(
        [sorted(team_ids), sorted(TARGET_SEASONS)],
        names=['team_id','season']
    ).to_frame(index=False)
    result = grid.merge(result, on=['team_id','season'], how='left', validate='one_to_one')
    missing = result.loc[result['continuity'].isna(), ['team_id','season']]
    if not missing.empty:
        print(f'WARNING: missing player minutes for {len(missing)} team-seasons. Continuity left blank.')
        print(missing.to_string(index=False))

    # Keep existing wins, if separately collected earlier; never overwrite them.
    if OUTPUT.exists():
        old = pd.read_csv(OUTPUT, dtype={'team_id':'string','season':'string'})
        if {'team_id','season','wins'}.issubset(old.columns):
            if old.duplicated(['team_id','season']).any():
                fail('Existing outcomes file contains duplicate team-season keys')
            result = result.merge(old[['team_id','season','wins']],
                                  on=['team_id','season'], how='left', validate='one_to_one')
    if 'wins' not in result.columns:
        result['wins'] = float('nan')

    # Preserve a concise outcomes file; continuity diagnostic columns can be
    # added separately if you want to audit individual team-season calculations.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result[['team_id','season','continuity','wins']].to_csv(OUTPUT, index=False)
    print('\nROSTER CONTINUITY CREATED')
    print('-------------------------')
    print('Total team-seasons:', len(result))
    print('Observed continuity:', result['continuity'].notna().sum())
    print('Missing continuity:', result['continuity'].isna().sum())
    print('Mean continuity (observed):', round(result['continuity'].mean(), 4))
    print('Saved to:', OUTPUT)
    print('\nNEXT: python src/construct/build_team_season_panel.py')

if __name__ == '__main__':
    main()
