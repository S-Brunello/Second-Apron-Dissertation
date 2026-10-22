
"""Collect structured 2024-25 SalarySwish contracts.

Uses the validated contract parser and the corrected
80-player URL audit. Does not overwrite existing data.
"""

from pathlib import Path
import time
import pandas as pd

from test_salaryswish_contract_parser import parse_contracts

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data/processed/"
    "salaryswish_2025_audit_corrected.csv"
)

OUTPUT = (
    ROOT / "data/processed/"
    "player_contracts_2025_v2.csv"
)

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}"
    )

audit = pd.read_csv(
    INPUT,
    dtype={"player_id": "string"}
)

if len(audit) != 80:
    raise ValueError(
        f"Expected 80 players, found {len(audit)}"
    )

if audit["player_id"].duplicated().any():
    raise ValueError("Duplicate player IDs in audit")

records = []
failures = []

for i, row in enumerate(audit.itertuples(index=False), 1):
    name = row.name
    url = row.source_url

    try:
        contracts = parse_contracts(url)

        if not contracts:
            failures.append({
                "player_id": row.player_id,
                "name": name,
                "reason": "No 2024-25 contract extracted"
            })

        for contract_number, contract in enumerate(
            contracts, start=1
        ):
            records.append({
                "player_id": row.player_id,
                "name": name,
                "season": 2025,
                "contract_number": contract_number,
                "contract_type": contract["contract_type"],
                "signing_team": contract["signing_team"],
                "signing_date": contract["signing_date"],
                "base_salary": contract["base_salary"],
                "cap_hit": contract["cap_hit"],
                "waiver_details": contract["waiver_details"],
                "conversion_details": contract[
                    "conversion_details"
                ],
                "source_url": url,
                "verification_status": "pending_review",
            })

        print(
            f"[{i:02d}/80] {name}: "
            f"{len(contracts)} contract(s)"
        )

    except Exception as error:
        failures.append({
            "player_id": row.player_id,
            "name": name,
            "reason": str(error)
        })
        print(f"[{i:02d}/80] {name}: ERROR - {error}")

    time.sleep(0.5)

if not records:
    raise ValueError("No contract records extracted")

df = pd.DataFrame(records)

# Preserve dates as distinct fields for later validation.
df["signing_date"] = pd.to_datetime(
    df["signing_date"], errors="coerce"
)

# Extract waiver date only when available.
waiver_date_text = df["waiver_details"].astype(
    "string"
).str.extract(
    r" - ([A-Za-z]+ \d{1,2}, \d{4})$"
)[0]

df["waiver_date"] = pd.to_datetime(
    waiver_date_text, errors="coerce"
)

# Flag two-way contracts without guessing for
# other contract categories.
df["is_two_way_contract"] = (
    df["contract_type"]
    .str.upper()
    .str.startswith("TWO-WAY CONTRACT", na=False)
)

# Flag contract histories that require attention.
df["review_flag"] = (
    df["signing_date"].isna()
    | df["contract_type"].eq("UNKNOWN")
    | df["base_salary"].isna()
)

# Save the raw extracted results, not a final
# game-by-game classification.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("2024-25 SALARYSWISH CONTRACT SUMMARY")
print("=" * 60)

print("\nPlayers in original audit:", len(audit))
print("Players with extracted contracts:",
      df["player_id"].nunique())
print("Total contract records:", len(df))

print("\nCONTRACT TYPES")
print(df["contract_type"].value_counts().to_string())

print("\nTWO-WAY CONTRACTS")
print("Records:", int(df["is_two_way_contract"].sum()))
print(
    "Unique players:",
    df.loc[df["is_two_way_contract"], "player_id"].nunique()
)

print("\nMULTIPLE CONTRACTS")
multiple = df.groupby(
    ["player_id", "name"]
).size()

print(
    multiple[multiple > 1]
    .sort_values(ascending=False)
    .to_string()
    if (multiple > 1).any()
    else "None"
)

print("\nRECORDS FLAGGED FOR REVIEW")
print(int(df["review_flag"].sum()))

print("\nEXTRACTION FAILURES")
if failures:
    print(pd.DataFrame(failures).to_string(index=False))
else:
    print("None")

print("\nSaved:", OUTPUT)
