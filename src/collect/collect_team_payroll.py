import pandas as pd

# SOURCE

# SalarySwish historical team payroll data
# 2023 page = 2022-23 NBA season
url = "https://www.salaryswish.com/past-team-payrolls/2023"

print("Downloading 2022-23 team payroll data...")

# READ TABLES FROM WEBPAGE

tables = pd.read_html(url)

print(f"Found {len(tables)} table(s) on the page.")

# Print information about each table so we can identify
# the correct payroll table.
for i, table in enumerate(tables):
    print(f"\n--- TABLE {i} ---")
    print("Columns:")
    print(table.columns)
    print("\nFirst five rows:")
    print(table.head())

print("\nFinished.")
