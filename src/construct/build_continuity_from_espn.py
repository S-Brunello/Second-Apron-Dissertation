"""Build provisional NBA roster continuity from ESPN player-game box scores.

INPUT: data/raw/source_tests/espn_player_box.parquet
OUTPUT: data/raw/team_season_outcomes.csv

DEFINITION:
  A player is returning to team j in season t only if the player recorded
  POSITIVE minutes for the SAME team in season t-1 and season t.
  - Leaving and returning within t still counts if present in t-1.
  - Returning after missing the entire previous team-season does NOT count.
  - Traded players' minutes belong to the actual team for each game.
  - This measures consecutive-season playing-time overlap, NOT unbroken
    tenure or accumulated teammate experience.

This deliberately leaves 2018-19 missing (no 2017-18 baseline) and 2025-26
missing (source has only 1,228 of 1,230 regular-season games).
It does NOT replace src/construct/build_roster_continuity.py; that script
requires all nine source seasons and is intended for the final full dataset.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/raw/source_tests/espn_player_box.parquet'
EXPOSURE = ROOT / 'data/processed/team_exposure.csv'
OUT = ROOT / 'data/raw/team_season_outcomes.csv'

EXPECTED_GAMES = {2019: 1230, 2020: 971, 2021: 1080, 2022: 1230,
                  2023: 1230, 2024: 1230, 2025: 1230}
# ESPN uses some historical abbreviations that differ from the canonical
# identifiers in our payroll exposure dataset.

ALIASES = {
    'BRK': 'BKN',
    'PHO': 'PHX',
    'GS': 'GSW',
    'NY': 'NYK',
    'SA': 'SAS',
    'SAN': 'SAS',
    'NO': 'NOP',
    'NOR': 'NOP',
    'UTH': 'UTA'
}



def parse_minutes(series):
    """Convert MM:SS to decimal minutes, allowing rounded :60 values."""
    values = series.astype("string").str.strip()

    # Accept seconds from 00 through 60.
    parts = values.str.extract(r"^(\d+):(\d{2})$")

    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce")

    invalid = (
        (values.notna() & parts[0].isna())
        | seconds.gt(60).fillna(False)
    )

    if invalid.any():
        raise ValueError(
            f"Unrecognized minutes values: "
            f"{values[invalid].unique()[:10].tolist()}"
        )

    rounded = seconds.eq(60).sum()
    print(f"Normalized {rounded} minute values ending in :60.")

    return minutes + seconds / 60


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing source file: {SOURCE}')
    if not EXPOSURE.exists():
        raise FileNotFoundError(f'Missing reference file: {EXPOSURE}')

    exposure = pd.read_csv(EXPOSURE, dtype={'team_id':'string','season':'string'})
    reference_teams = set(exposure.loc[exposure.season.eq('2022-23'), 'team_id'])
    if len(reference_teams) != 30:
        raise ValueError('Expected 30 canonical teams in exposure file')

    raw = pd.read_parquet(SOURCE)
    needed = {'season', 'game_id', 'player_id', 'team', 'minutes_played'}
    if missing := (needed - set(raw.columns)):
        raise ValueError(f'Missing ESPN columns: {sorted(missing)}')

    # NBA game IDs starting 002 denote regular season games.
    df = raw.loc[raw.game_id.astype(str).str.startswith('002'),
                 list(needed)].copy()
    df['season'] = pd.to_numeric(df['season'], errors='raise').astype(int)
    if df.duplicated(['season', 'game_id', 'player_id', 'team']).any():
        raise ValueError('Duplicate player-game-team rows detected')

    counts = df.groupby('season')['game_id'].nunique()
    for year, expected in EXPECTED_GAMES.items():
        actual = int(counts.get(year, 0))
        if actual != expected:
            raise ValueError(f'Incomplete season ending {year}: {actual}/{expected} games')
    print('Validated complete regular seasons ending 2019 through 2025.')
    print('2025-26 excluded: source has 1,228/1,230 regular-season games.')

    df = df[df.season.between(2019, 2025)].copy()
    df['team_id'] = df['team'].replace(ALIASES)
    unknown = set(df.team_id.dropna()) - reference_teams
    if unknown or df.team_id.isna().any():
        raise ValueError(f'Unmapped ESPN team identifiers: {sorted(unknown)}')
    df['minutes'] = parse_minutes(df.minutes_played)
    # DNP rows contribute no minutes; retaining them must not mark roster tenure.
    df = df.loc[df.minutes.gt(0)].copy()
    df['player_id'] = df.player_id.astype('string')

    player_seasons = (df.groupby(['season','team_id','player_id'], as_index=False)
                      .agg(minutes=('minutes', 'sum')))
    player_seasons['season'] = player_seasons['season'].map(
        lambda y: f'{y - 1}-{str(y)[-2:]}')

    # Save source-compatible player-team-season records for future audits.
    minutes_out = ROOT / 'data/raw/player_team_season_minutes.csv'
    minutes_out.parent.mkdir(parents=True, exist_ok=True)
    player_seasons[['player_id','team_id','season','minutes']].to_csv(minutes_out, index=False)

    previous = player_seasons[['player_id','team_id','season']].copy()
    previous['season_start'] = previous.season.str[:4].astype(int) + 1
    previous = previous[['player_id','team_id','season_start']].assign(returning=True)
    current = player_seasons.copy()
    current['season_start'] = current.season.str[:4].astype(int)
    current = current.merge(previous, on=['player_id','team_id','season_start'],
                            how='left', validate='one_to_one')
    current['returning'] = current.returning.fillna(False).astype(bool)
    current['returning_minutes'] = current.minutes.where(current.returning, 0)

    # First computable season is 2019-20, since 2018-19 is the baseline.
    current = current[current.season_start.between(2019, 2024)]
    result = (current.groupby(['team_id','season'], as_index=False)
              .agg(total_minutes=('minutes','sum'),
                   returning_minutes=('returning_minutes','sum')))
    result['continuity'] = result.returning_minutes / result.total_minutes
    if len(result) != 180 or not result.continuity.between(0, 1).all():
        raise ValueError('Expected 180 valid 2019-20 through 2024-25 continuity observations')

    # Preserve any existing wins; preserve other existing continuity for
    # seasons outside the provisional six-season window, if present.
    grid = pd.MultiIndex.from_product(
        [sorted(reference_teams),
         [f'{y}-{str(y+1)[-2:]}' for y in range(2018, 2026)]],
        names=['team_id','season']).to_frame(index=False)
    final = grid.merge(result[['team_id','season','continuity']],
                       on=['team_id','season'], how='left', validate='one_to_one')
    if OUT.exists():
        old = pd.read_csv(OUT, dtype={'team_id':'string','season':'string'})
        if old.duplicated(['team_id','season']).any():
            raise ValueError('Existing outcomes file has duplicate team-season keys')
        cols = ['team_id','season'] + (['wins'] if 'wins' in old.columns else [])
        final = final.merge(old[cols], on=['team_id','season'], how='left', validate='one_to_one')
    if 'wins' not in final.columns:
        final['wins'] = float('nan')
    final.to_csv(OUT, index=False)
    print('Player-team-season records:', len(player_seasons))
    print('Continuity observations:', final.continuity.notna().sum())
    print('Missing continuity observations:', final.continuity.isna().sum())
    print('Saved:', minutes_out)
    print('Saved:', OUT)
    print('NEXT: python src/construct/build_team_season_panel.py')


if __name__ == '__main__':
    main()
