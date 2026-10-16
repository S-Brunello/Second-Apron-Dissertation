#Build a 'game ID to date' using llimllib/nba_data.

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/source_tests/espn_player_box.parquet"
DEST = ROOT / "data/raw/game_dates.csv"

OWNER = "llimllib"
REPO = "nba_data"
BRANCH = "main"

# Existing regular-season game IDs
df = pd.read_parquet(SOURCE)
games = df[["season", "game_id"]].copy()
games["game_id"] = games["game_id"].astype(str)
games = games[games["game_id"].str.startswith("002")]
games = games.drop_duplicates()

wanted = set(games["game_id"])

print(f"Regular-season games to match: {len(wanted):,}")

# List upstream files once using GitHub's tree API
tree_url = (
    f"https://api.github.com/repos/{OWNER}/{REPO}"
    f"/git/trees/{BRANCH}?recursive=1"
)

response = requests.get(tree_url, timeout=60)
response.raise_for_status()
tree_data = response.json()

if tree_data.get("truncated"):
    raise RuntimeError("GitHub file listing was truncated.")

# Match date-named JSON files for our seasons
seasons = set(games["season"].astype(int))

pattern = re.compile(
    r"^data/espn/(\d{4})/(\d{4}-\d{2}-\d{2})\.json$"
)

files = []

for item in tree_data["tree"]:
    match = pattern.match(item["path"])

    if match and int(match.group(1)) in seasons:
        files.append((item["path"], match.group(2)))

print(f"Date-specific JSON files found: {len(files):,}")

if not files:
    raise RuntimeError("No matching JSON files found.")

def read_game_date(item):
    path, date = item

    url = (
        f"https://raw.githubusercontent.com/"
        f"{OWNER}/{REPO}/{BRANCH}/{path}"
    )

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=40)
            response.raise_for_status()

            data = response.json()
            rows = data.get("player_box", [])

            ids = {
                str(row["gmId"])
                for row in rows
                if row.get("gmId") is not None
            }

            return [
                {"game_id": game_id, "game_date": date}
                for game_id in ids
                if game_id in wanted
            ]

        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

records = []
errors = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(read_game_date, item): item
        for item in files
    }

    for count, future in enumerate(
        as_completed(futures), start=1
    ):
        try:
            records.extend(future.result())
        except Exception as error:
            errors.append((futures[future][0], str(error)))

        if count % 100 == 0:
            print(f"Processed {count}/{len(files)} files")

if not records:
    raise RuntimeError("No game-date matches found.")

lookup = pd.DataFrame(records).drop_duplicates()

# Check for conflicting dates
conflicts = (
    lookup.groupby("game_id")["game_date"]
    .nunique()
)

if (conflicts > 1).any():
    raise RuntimeError(
        "Some game IDs have conflicting dates. "
        "Review before saving."
    )

lookup = lookup.drop_duplicates("game_id")

# Merge to recover and validate the season
lookup = games.merge(
    lookup, on="game_id", how="left",
    validate="one_to_one"
)

matched = lookup["game_date"].notna().sum()
total = len(lookup)

print("\nVALIDATION REPORT")
print(f"Expected games: {total:,}")
print(f"Matched games: {matched:,}")
print(f"Unmatched games: {total - matched:,}")
print(f"Coverage: {matched / total:.2%}")
print(f"Download errors: {len(errors)}")

print("\nCoverage by season:")
print(
    lookup.groupby("season")["game_date"]
    .agg(total="size", matched="count")
    .to_string()
)

if errors:
    print("\nFirst 10 download errors:")
    for error in errors[:10]:
        print(error)

if matched != total:
    print("\nFirst 20 unmatched games:")
    print(
        lookup[lookup["game_date"].isna()]
        .head(20)
        .to_string(index=False)
    )

# Do not save an incomplete lookup.
if matched != total or errors:
    print("\nNOT SAVED: Coverage requires investigation.")
else:
    lookup["game_date"] = pd.to_datetime(
        lookup["game_date"]
    ).dt.strftime("%Y-%m-%d")

    lookup = lookup[
        ["game_id", "game_date", "season"]
    ].sort_values(["season", "game_date", "game_id"])

    if DEST.exists():
        print(f"\nNOT OVERWRITTEN: {DEST} already exists.")
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        lookup.to_csv(DEST, index=False)
        print(f"\nSaved: {DEST}")
