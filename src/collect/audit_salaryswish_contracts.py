
from pathlib import Path
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/salaryswish_2025_audit.csv"

if OUT.exists():
    raise FileExistsError(f"Review existing output first: {OUT}")

espn = pd.read_parquet(
    ROOT / "data/raw/source_tests/espn_player_box.parquet"
)
salary = pd.read_parquet(
    ROOT / "data/raw/historical_player_salaries.parquet"
)

season = 2025

espn = espn[
    (espn["season"] == season)
    & espn["game_id"].astype(str).str.startswith("002")
].copy()

espn["player_id"] = espn["player_id"].astype("string")
salary["player_id"] = salary["player_id"].astype("string")

players = espn[
    ["player_id", "name"]
].drop_duplicates("player_id")

valid_ids = set(
    salary.loc[
        (salary["season"] == season)
        & salary["salary"].notna(),
        "player_id"
    ].dropna()
)

missing = players[
    ~players["player_id"].isin(valid_ids)
].copy()

assert len(missing) == 80, (
    f"Expected 80 missing-salary players; got {len(missing)}"
)

def slugify(name):
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        c for c in name
        if not unicodedata.combining(c)
    )
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

records = []

for i, row in enumerate(missing.itertuples(index=False), 1):
    name = row.name
    url = (
        "https://www.salaryswish.com/players/"
        + slugify(name)
    )

    record = {
        "player_id": row.player_id,
        "name": name,
        "source_url": url,
        "http_status": None,
        "2024_25_contract_excerpt": "",
        "audit_status": "unreviewed",
    }

    try:
        response = session.get(url, timeout=25)
        record["http_status"] = response.status_code

        if response.status_code != 200:
            record["audit_status"] = "page_unavailable"
        else:
            soup = BeautifulSoup(
                response.text, "html.parser"
            )
            text = soup.get_text(" ", strip=True)

            start = text.upper().find("CONTRACT HISTORY")

            if start < 0:
                record["audit_status"] = "no_contract_history"
            else:
                history = text[start:]

                # Locate each contract block using the
                # repeated "COMPARE THIS CONTRACT" heading.
                headings = list(re.finditer(
                    r"COMPARE THIS CONTRACT",
                    history,
                    flags=re.I
                ))

                matches = []

                for j, heading in enumerate(headings):
                    next_start = (
                        headings[j + 1].start()
                        if j + 1 < len(headings)
                        else len(history)
                    )

                    # Include preceding contract title.
                    block_start = max(0, heading.start() - 65)
                    block = history[block_start:next_start]

                    # Extract only contract blocks that
                    # explicitly contain the 2024-25 season.
                    if re.search(r"\b2024-25\b", block):
                        matches.append(block[:1400])

                if matches:
                    record["2024_25_contract_excerpt"] = (
                        "\n--- CONTRACT BLOCK ---\n".join(matches)
                    )
                    record["audit_status"] = "candidate_found"
                else:
                    record["audit_status"] = "no_2024_25_block"

    except requests.RequestException as error:
        record["audit_status"] = "request_error"
        record["2024_25_contract_excerpt"] = str(error)

    records.append(record)

    print(
        f"[{i:02d}/80] {name}: "
        f"{record['audit_status']}"
    )

    time.sleep(0.5)

df = pd.DataFrame(records)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("\nSALARYSWISH AUDIT SUMMARY")
print(df["audit_status"].value_counts().to_string())

print("\nPLAYERS WITHOUT A CANDIDATE CONTRACT")
print(
    df.loc[
        df["audit_status"] != "candidate_found",
        ["name", "audit_status", "source_url"]
    ].to_string(index=False)
)

print("\nSaved:", OUT)
