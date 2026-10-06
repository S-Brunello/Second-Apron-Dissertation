import pandas as pd
from pathlib import Path
from io import StringIO

project_root = Path(__file__).resolve().parents[2]

input_file = (
    project_root
    / "data/raw/roster_continuity/continuity_bref_raw.html"
)

html = input_file.read_text(encoding="utf-8")

# Some source tables may be inside HTML comments.
# Remove comment markers in memory; preserve the original file.
parse_html = html.replace("<!--", "").replace("-->", "")

tables = pd.read_html(StringIO(parse_html))

print(f"Tables found: {len(tables)}")

for i, table in enumerate(tables):
    print(f"\nTABLE {i}")
    print("Shape:", table.shape)
    print("Columns:", table.columns.tolist())
    print(table.head().to_string(index=False))

    # Select the continuity table by its column headings
matches = [
    table for table in tables
    if "Season" in table.columns
    and {"ATL", "BOS", "NJN", "NOH", "PHO"}.issubset(table.columns)
]

assert len(matches) == 1, "Expected exactly one continuity table."
wide = matches[0].copy()

# Keep the dissertation sample
sample_seasons = [
    f"{year}-{str(year + 1)[-2:]}"
    for year in range(2018, 2026)
]

wide["Season"] = wide["Season"].astype(str).str.strip()

wide = wide[
    wide["Season"].isin(sample_seasons)
].copy()

assert wide["Season"].is_unique, "Duplicate seasons found."
assert set(wide["Season"]) == set(sample_seasons), \
    "Some sample seasons are missing."

# Convert team columns into team-season rows
long = wide.melt(
    id_vars="Season",
    var_name="source_team_id",
    value_name="continuity_pct_raw"
)

long = long.rename(columns={"Season": "season"})

# Match franchise labels to our existing team IDs
long["team_id"] = long["source_team_id"].replace({
    "NJN": "BKN",
    "NOH": "NOP",
    "PHO": "PHX"
})

# Convert percentage strings, such as '75%', into 0.75
long["continuity"] = pd.to_numeric(
    long["continuity_pct_raw"]
        .astype(str)
        .str.strip()
        .str.removesuffix("%"),
    errors="raise"
) / 100

# Validate against the repository's team registry
teams = pd.read_csv(project_root / "data/raw/teams.csv")

assert set(long["team_id"]) == set(teams["team_id"]), \
    "Continuity team IDs do not match teams.csv."

assert len(long) == 240, "Expected 30 teams × 8 seasons."
assert not long.duplicated(["team_id", "season"]).any(), \
    "Duplicate team-season observations."

assert long["continuity"].notna().all(), \
    "Missing continuity values."

assert long["continuity"].between(0, 1).all(), \
    "Continuity must be between 0 and 1."

# Keep source values alongside the cleaned measure
clean = long[
    [
        "team_id",
        "season",
        "continuity",
        "source_team_id",
        "continuity_pct_raw"
    ]
].sort_values(["team_id", "season"]).reset_index(drop=True)

# Save
output_folder = project_root / "data/processed"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "team_continuity.csv"
clean.to_csv(output_file, index=False)

print("\nCleaned data:")
print(clean.head(10).to_string(index=False))

print("\nTeams per season:")
print(clean.groupby("season")["team_id"].nunique())

print(f"\nSaved {len(clean)} observations to: {output_file}")