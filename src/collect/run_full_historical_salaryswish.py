
"""Collect SalarySwish contracts for all historical target players.

Preserves existing files and saves progress after each player.
"""

from pathlib import Path
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_historical_salaryswish_contracts import parse_contracts

ROOT = Path(__file__).resolve().parents[2]

TARGETS = ROOT / "data/processed/historical_salaryswish_targets.csv"

OUTPUT = ROOT / "data/processed/historical_salaryswish_contracts_full.csv"
AUDIT = ROOT / "data/processed/historical_salaryswish_audit_full.csv"

if OUTPUT.exists() or AUDIT.exists():
    raise FileExistsError(
        "Full extraction files already exist. "
        "Do not overwrite them without checking."
    )

targets = pd.read_csv(TARGETS, dtype={"player_id": "string"})

assert len(targets) == 230
assert not targets.duplicated(["season", "player_id"]).any()

contracts = []
audit = []

for i, row in enumerate(targets.itertuples(index=False), start=1):

    nba_season = f"{row.season - 1}-{str(row.season)[-2:]}"

    status = "success"
    error = ""
    extracted = []

    try:
        extracted = parse_contracts(
            row.source_url,
            nba_season
        )

        if not extracted:
            status = "no_season_contract"

        for contract in extracted:
            contracts.append({
                "season": row.season,
                "nba_season": nba_season,
                "player_id": row.player_id,
                "name": row.name,
                "source_url": row.source_url,
                **contract,
            })

    except (requests.RequestException, ValueError) as exc:
        status = "error"
        error = str(exc)[:250]

    audit.append({
        "season": row.season,
        "player_id": row.player_id,
        "name": row.name,
        "status": status,
        "contracts_found": len(extracted),
        "error": error,
        "source_url": row.source_url,
    })

    # Save progress after each completed player.
    pd.DataFrame(contracts).to_csv(OUTPUT, index=False)
    pd.DataFrame(audit).to_csv(AUDIT, index=False)

    print(
        f"[{i}/{len(targets)}] "
        f"{nba_season} | {row.name} | "
        f"{status} | {len(extracted)} contracts",
        flush=True
    )

    time.sleep(0.5)

print("\nFULL HISTORICAL EXTRACTION SUMMARY")
print(pd.DataFrame(audit)["status"].value_counts().to_string())

print("\nTotal contracts:", len(contracts))

print("\nContracts by season:")
if contracts:
    print(
        pd.DataFrame(contracts)
        .groupby("nba_season")
        .size()
        .to_string()
    )

print("\nSaved:", OUTPUT)
print("Audit:", AUDIT)
