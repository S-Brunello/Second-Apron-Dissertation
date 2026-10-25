
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/historical_salaryswish_contracts_v2.csv"
OUTPUT = ROOT / "data/processed/historical_contract_date_audit.csv"

if OUTPUT.exists():
    raise FileExistsError(f"Already exists: {OUTPUT}")

df = pd.read_csv(SOURCE, dtype={"player_id": "string"})

df["signed"] = pd.to_datetime(
    df["signing_date"], errors="coerce"
)

for field in ["waiver_details", "conversion_details"]:
    extracted = df[field].astype("string").str.extract(
        r"([A-Za-z]{3,9}\s+\d{1,2},?\s+20\d{2})",
        expand=False
    )
    df[field + "_date"] = pd.to_datetime(
        extracted, errors="coerce"
    )

df["date_issue"] = "none"

missing_waiver = (
    df["waiver_details"].notna()
    & df["waiver_details_date"].isna()
)

backwards_waiver = (
    df["waiver_details_date"].notna()
    & (df["waiver_details_date"] < df["signed"])
)

backwards_conversion = (
    df["conversion_details_date"].notna()
    & (df["conversion_details_date"] < df["signed"])
)

df.loc[missing_waiver, "date_issue"] = "missing_waiver_date"
df.loc[backwards_waiver, "date_issue"] = "waiver_before_signing"
df.loc[backwards_conversion, "date_issue"] = "conversion_before_signing"

issues = df[df["date_issue"] != "none"].copy()

issues[
    [
        "season", "player_id", "name", "signing_team",
        "contract_type", "signing_date",
        "waiver_details", "conversion_details",
        "date_issue", "source_url"
    ]
].to_csv(OUTPUT, index=False)

print("\nHISTORICAL CONTRACT DATE EXCEPTIONS")
print(
    issues[
        ["nba_season", "name", "contract_type",
         "signing_date", "waiver_details", "date_issue"]
    ].to_string(index=False)
)

print("\nException contract records:", len(issues))
print("Unique player-season cases:",
      issues[["season", "player_id"]].drop_duplicates().shape[0])
print("\nSaved:", OUTPUT)
