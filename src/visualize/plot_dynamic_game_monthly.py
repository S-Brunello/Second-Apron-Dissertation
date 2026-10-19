
"""Plot monthly dynamic exposure estimates with 95% confidence intervals."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/tables/dynamic_game_monthly.csv"
OUTPUT = ROOT / "figures/dynamic_game_monthly.png"

df = pd.read_csv(SOURCE)

# The omitted reference period has coefficient zero by definition.
reference = df["reference"].iloc[0]

ref_row = pd.DataFrame([{
    "period": reference,
    "coefficient_pp": 0.0,
    "ci_lower": 0.0,
    "ci_upper": 0.0
}])

df = pd.concat([df, ref_row], ignore_index=True)
df = df.sort_values("period").reset_index(drop=True)

df["position"] = range(len(df))

fig, ax = plt.subplots(figsize=(15, 6.5))

# Plot non-reference coefficients and confidence intervals.
estimates = df[df["period"] != reference]

ax.errorbar(
    estimates["position"],
    estimates["coefficient_pp"],
    yerr=[
        estimates["coefficient_pp"] - estimates["ci_lower"],
        estimates["ci_upper"] - estimates["coefficient_pp"]
    ],
    fmt="o",
    markersize=3.5,
    capsize=2,
    linewidth=1,
    label="Estimate and 95% CI"
)

# Reference month: normalized to zero, not independently estimated.
ref_position = df.loc[
    df["period"] == reference, "position"
].iloc[0]

ax.scatter(
    [ref_position], [0],
    marker="s", s=45,
    label="Reference: March 2023"
)

ax.axhline(0, color="black", linewidth=1)

# Start of the first CBA season and later implementation phase.
for period, label, style in [
    ("2023-10", "2023-24 season", "--"),
    ("2024-10", "2024-25 season", ":")
]:
    match = df.loc[df["period"] == period, "position"]

    if not match.empty:
        ax.axvline(
            match.iloc[0] - 0.5,
            linestyle=style,
            color="gray",
            linewidth=1.4,
            label=label
        )

ax.set_xticks(df["position"][::2])
ax.set_xticklabels(
    df["period"][::2],
    rotation=65,
    ha="right"
)

ax.set_title(
    "Dynamic Payroll Exposure and Roster Continuity: Monthly"
)

ax.set_xlabel("Observed NBA Calendar Month")
ax.set_ylabel(
    "Coefficient per 0.10 Higher Exposure (Percentage Points)"
)

ax.grid(axis="y", alpha=0.2)
ax.legend(loc="best")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300)
plt.close()

print("Saved:", OUTPUT)
print("Periods plotted:", len(df))
print("Reference:", reference)
