
from pathlib import Path
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

targets = pd.read_csv(
    ROOT / "data/processed/historical_salaryswish_targets.csv",
    dtype={"player_id": "string"}
)

sample = targets[
    targets["season"].isin([2023, 2024])
    & targets["url_source"].eq("generated_unverified")
].head(20).copy()

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

results = []

for row in sample.itertuples():
    try:
        response = session.get(
            row.source_url,
            timeout=20
        )

        text = response.text.lower()

        has_contracts = (
            "compare this contract" in text
            or "two-way contract" in text
            or "rookie contract" in text
            or "veteran contract" in text
        )

        status = (
            "usable_page"
            if response.status_code == 200 and has_contracts
            else "needs_review"
        )

        http_code = response.status_code

    except requests.RequestException:
        status = "request_failed"
        http_code = None

    results.append({
        "season": row.season,
        "name": row.name,
        "http_code": http_code,
        "page_status": status,
        "source_url": row.source_url
    })

    time.sleep(0.5)

results = pd.DataFrame(results)

print("\nSALARYSWISH URL TEST")
print(
    results[
        ["season", "name", "http_code", "page_status"]
    ].to_string(index=False)
)

print("\nSUMMARY")
print(results["page_status"].value_counts().to_string())

print("\nNo files changed.")
