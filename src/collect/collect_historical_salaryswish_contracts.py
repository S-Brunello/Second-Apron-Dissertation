
"""Extract historical SalarySwish contracts by NBA season.

Runs a small pilot for 2022-23 and 2023-24.
Does not alter the existing 2024-25 files.
"""

from pathlib import Path
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]

TARGETS = (
    ROOT / "data/processed/historical_salaryswish_targets.csv"
)
OUTPUT = (
    ROOT / "data/processed/historical_salaryswish_contracts_pilot.csv"
)
AUDIT = (
    ROOT / "data/processed/historical_salaryswish_pilot_audit.csv"
)


def dollars(value):
    return int(value.replace("$", "").replace(",", ""))


def parse_contracts(url, season):
    """Extract contract records relevant to one NBA season."""

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(" ", strip=True)

    start = text.upper().find("CONTRACT HISTORY")

    if start < 0:
        raise ValueError("Contract history not found")

    history = text[start:]

    markers = list(
        re.finditer(r"COMPARE THIS CONTRACT", history, re.I)
    )

    if not markers:
        raise ValueError("No contract markers found")

    results = []

    for i, marker in enumerate(markers):
        end = (
            markers[i + 1].start()
            if i + 1 < len(markers)
            else len(history)
        )

        block = history[marker.end():end]

        preceding = history[
            max(0, marker.start() - 150):marker.start()
        ]

        
        # Contract type appears immediately before
        # "COMPARE THIS CONTRACT".
        # Check the longest, most specific patterns first.

        contract_patterns = [
            r"TWO-WAY CONTRACT \(REST OF SEASON\) \(EXTENSION\)",
            r"TWO-WAY CONTRACT \(REST OF SEASON\)",
            r"TWO-WAY CONTRACT",
            r"ROOKIE SCALE CONTRACT",
            r"ROOKIE CONTRACT \(REST OF SEASON\)",
            r"ROOKIE CONTRACT",
            r"VETERAN CONTRACT \(REST OF SEASON\)",
            r"VETERAN CONTRACT",
            r"MINIMUM SALARY CONTRACT",
            r"10-DAY CONTRACT",
            r"EXHIBIT 10",
            r"CAMP INVITE",
        ]

        contract_type = "UNKNOWN"

        for pattern in contract_patterns:
            match = re.search(
                rf"({pattern})"
                rf"(?:\s+(?:INCLUDES\b.*|38\+\s+RULE\b.*))?$",
                preceding.strip(),
                re.I,
            )

            if match:
                contract_type = match.group(1).upper()
                break

        # Remove page heading accidentally included in contract type.
        contract_type = re.sub(
            r"^CONTRACT HISTORY\s+",
            "",
            contract_type
        )

        # Flag historical contracts that still require review.
        if contract_type == "UNKNOWN":
            print(f"WARNING: Unknown contract type at {url}")

        table_start = block.find("Season Option")

        if table_start < 0:
            continue

        table_text = block[table_start:]

        total_match = re.search(r"\bTOTAL\b", table_text)

        if total_match:
            table_text = table_text[:total_match.start()]

        # The key change: use the requested season.
        season_row = re.search(
            rf"\b{re.escape(season)}\b"
            r"(?:(?!\b20\d\d-\d\d\b).){0,180}?"
            r"(\$[\d,]+)\s+(\$[\d,]+)",
            table_text,
            re.S,
        )

        if not season_row:
            continue

        team = re.search(
            r"Signing\s+Team\s*:\s*([A-Z]{2,3})",
            block,
            re.I,
        )

        signed = re.search(
            r"Signing\s+Date\s*:\s*"
            r"([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            block,
            re.I,
        )

        waiver = re.search(
            r"WAIVED:\s*This contract was waived"
            r"\s*\(([^)]*)\)",
            block,
            re.I,
        )

        conversion = re.search(
            r"VOIDED DUE TO CONVERSION:\s*"
            r"([^(]*\([^)]*\))",
            block,
            re.I,
        )

        results.append({
            "contract_type": contract_type,
            "signing_team": team.group(1) if team else None,
            "signing_date": signed.group(1) if signed else None,
            "cap_hit": dollars(season_row.group(1)),
            "base_salary": dollars(season_row.group(2)),
            "waiver_details": (
                waiver.group(1) if waiver else None
            ),
            "conversion_details": (
                conversion.group(1) if conversion else None
            ),
        })

    return results


def main():
    for path in [OUTPUT, AUDIT]:
        if path.exists():
            raise FileExistsError(
                f"Pilot file already exists: {path}"
            )

    targets = pd.read_csv(
        TARGETS,
        dtype={"player_id": "string"},
    )

    # Five player-season cases from each target season.
    pilot = pd.concat(
        [
            targets[targets["season"] == year].head(5)
            for year in [2023, 2024]
        ],
        ignore_index=True,
    )

    contracts = []
    audit = []

    for row in pilot.itertuples(index=False):
        nba_season = (
            f"{row.season - 1}-{str(row.season)[-2:]}"
        )

        status = "success"
        error = ""

        try:
            extracted = parse_contracts(
                row.source_url,
                nba_season,
            )

            if not extracted:
                status = "no_season_contract"

            for record in extracted:
                contracts.append({
                    "season": row.season,
                    "nba_season": nba_season,
                    "player_id": row.player_id,
                    "name": row.name,
                    "source_url": row.source_url,
                    **record,
                })

        except (requests.RequestException, ValueError) as exc:
            status = "error"
            error = str(exc)[:250]
            extracted = []

        audit.append({
            "season": row.season,
            "name": row.name,
            "status": status,
            "contracts_found": len(extracted),
            "error": error,
            "source_url": row.source_url,
        })

        print(
            f"{nba_season} | {row.name} | "
            f"{status} | {len(extracted)} contracts"
        )

        time.sleep(0.5)

    pd.DataFrame(contracts).to_csv(OUTPUT, index=False)
    pd.DataFrame(audit).to_csv(AUDIT, index=False)

    print("\nPILOT SUMMARY")
    print(pd.DataFrame(audit)["status"].value_counts())

    print("\nExtracted contracts:", len(contracts))
    print("Contracts saved:", OUTPUT)
    print("Audit saved:", AUDIT)


if __name__ == "__main__":
    main()
