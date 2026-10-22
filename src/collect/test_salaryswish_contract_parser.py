
import re
import requests
from bs4 import BeautifulSoup

PLAYERS = {
    "Alondes Williams":
        "https://www.salaryswish.com/players/alondes-williams",
    "Kessler Edwards":
        "https://www.salaryswish.com/players/kessler-edwards",
}

SEASON = "2024-25"


def dollars(value):
    return int(value.replace("$", "").replace(",", ""))


def parse_contracts(url):
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

    # Identify individual contract records.
    markers = list(re.finditer(
        r"COMPARE THIS CONTRACT",
        history,
        re.I
    ))

    results = []

    for i, marker in enumerate(markers):

        # Isolate this contract from the next one.
        end = (
            markers[i + 1].start()
            if i + 1 < len(markers)
            else len(history)
        )

        block = history[marker.end():end]

        # Identify contract type from text directly
        # preceding this contract's marker.
        preceding = history[
            max(0, marker.start() - 150):marker.start()
        ]

        heading_match = re.search(
            r"(TWO-WAY CONTRACT(?: \(REST OF SEASON\))?"
            r"|EXHIBIT 10"
            r"|ROOKIE SCALE CONTRACT"
            r"|ROOKIE CONTRACT(?: \(REST OF SEASON\))?"
            r"|VETERAN CONTRACT(?: \(REST OF SEASON\))?"
            r"|MINIMUM SALARY CONTRACT(?: \(REST OF SEASON\))?"
            r"|[A-Z][A-Z0-9 /()-]*CONTRACT"
            r"(?: \(REST OF SEASON\))?)\s*$",
            preceding,
            re.I
        )

        contract_type = (
            heading_match.group(1).upper().strip()
            if heading_match
            else "UNKNOWN"
        )

        # Extract only the salary table, excluding cap holds.
        table_start = block.find("Season Option")

        if table_start < 0:
            continue

        table_text = block[table_start:]

        total_match = re.search(
            r"\bTOTAL\b",
            table_text
        )

        if total_match:
            table_text = table_text[:total_match.start()]

        # Find the 2024-25 salary row.
        season_row = re.search(
            rf"\b{SEASON}\b"
            r"(?:(?!\b20\d\d-\d\d\b).){0,180}?"
            r"(\$[\d,]+)\s+(\$[\d,]+)",
            table_text,
            re.S
        )

        if not season_row:
            continue

        # Signing information
        team = re.search(
            r"Signing\s+Team\s*:\s*([A-Z]{2,3})",
            block,
            re.I
        )

        signed = re.search(
            r"Signing\s+Date\s*:\s*"
            r"([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            block,
            re.I
        )

        # Waiver and conversion information
        waiver = re.search(
            r"WAIVED:\s*This contract was waived"
            r"\s*\(([^)]*)\)",
            block,
            re.I
        )

        conversion = re.search(
            r"VOIDED DUE TO CONVERSION:\s*"
            r"([^(]*\([^)]*\))",
            block,
            re.I
        )

        results.append({
            "contract_type": contract_type,
            "signing_team": (
                team.group(1) if team else None
            ),
            "signing_date": (
                signed.group(1) if signed else None
            ),
            "base_salary": dollars(
                season_row.group(2)
            ),
            "cap_hit": dollars(
                season_row.group(1)
            ),
            "waiver_details": (
                waiver.group(1) if waiver else None
            ),
            "conversion_details": (
                conversion.group(1)
                if conversion else None
            ),
        })

    return results


if __name__ == "__main__":

    for name, url in PLAYERS.items():
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        contracts = parse_contracts(url)

        if not contracts:
            print("NO 2024-25 CONTRACTS EXTRACTED")

        for i, contract in enumerate(contracts, 1):
            print(f"\nContract {i}")

            for field, value in contract.items():
                print(f"  {field}: {value}")

    print("\nNo files changed.")
