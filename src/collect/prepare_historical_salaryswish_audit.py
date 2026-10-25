
#Prepare historical missing-salary players for SalarySwish recovery.

from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CLASSIFICATION = ROOT / "data/processed/player_salary_classification.csv"
PREVIOUS = ROOT / "data/processed/salaryswish_2025_audit_corrected.csv"
OUTPUT = ROOT / "data/processed/historical_salaryswish_targets.csv"

if OUTPUT.exists():
    raise FileExistsError(f"Already exists: {OUTPUT}")

df = pd.read_csv(CLASSIFICATION, dtype={"player_id": "string"})
previous = pd.read_csv(PREVIOUS, dtype={"player_id": "string"})

targets = (
    df[
        df["salary_source"].eq("unresolved")
        & df["season"].ne(2025)
    ][["season", "player_id", "name"]]
    .drop_duplicates(["season", "player_id"])
    .copy()
)

def slugify(name):
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    return text.strip("-")

known_urls = (
    previous[["player_id", "source_url"]]
    .drop_duplicates("player_id")
    .rename(columns={"source_url": "known_url"})
)

targets = targets.merge(
    known_urls,
    on="player_id",
    how="left",
    validate="many_to_one"
)

targets["source_url"] = targets["known_url"].fillna(
    "https://www.salaryswish.com/players/"
    + targets["name"].apply(slugify)
)

targets["url_source"] = targets["known_url"].apply(
    lambda x: "previously_verified" if pd.notna(x)
    else "generated_unverified"
)

targets["nba_season"] = (
    (targets["season"] - 1).astype(str)
    + "-"
    + targets["season"].astype(str).str[-2:]
)

targets = targets.drop(columns="known_url").sort_values(
    ["season", "name"]
)

targets.to_csv(OUTPUT, index=False)

print("\nHISTORICAL SALARYSWISH RECOVERY TARGETS")
print(
    targets.groupby(["nba_season", "url_source"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nTotal player-season cases:", len(targets))
print("Unique player IDs:", targets["player_id"].nunique())
print("Previously verified URLs:",
      (targets["url_source"] == "previously_verified").sum())
print("Generated URLs requiring validation:",
      (targets["url_source"] == "generated_unverified").sum())

print("\nSaved:", OUTPUT)
