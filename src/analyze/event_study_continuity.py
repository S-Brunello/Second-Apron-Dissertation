
"""
Continuous-exposure event study: roster continuity

Outcome:
    Share of team minutes played by returning players.

Exposure:
    Fixed 2022-23 payroll-to-salary-cap ratio.

Specification:
    Team FE + season FE + season × exposure interactions.

Reference season:
    2022-23 (last pre-CBA season).

Inference:
    Standard errors clustered by NBA team.

Outputs:
    figures/continuity_event_study.png
    data/processed/continuity_event_study_results.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/team_season_panel.csv"

FIGURE = ROOT / "figures/continuity_event_study.png"
RESULTS = (
    ROOT / "data/processed/"
    "continuity_event_study_results.csv"
)

REFERENCE = "2022-23"

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

# 0.10 corresponds to ten percentage points
# of the salary cap in payroll exposure.
EXPOSURE_STEP = 0.10


def main():
    df = pd.read_csv(INPUT)

    required = [
        "team_id", "season", "exposure", "continuity"
    ]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Only use observations with measured continuity.
    df = df.dropna(
        subset=["continuity", "exposure"]
    ).copy()

    df["season"] = df["season"].astype(str)

    if set(df["season"]) != set(SEASONS):
        raise ValueError(
            "Unexpected season coverage: "
            f"{sorted(df['season'].unique())}"
        )

    if df.duplicated(["team_id", "season"]).any():
        raise ValueError("Duplicate team-season records.")

    if len(df) != 180 or df["team_id"].nunique() != 30:
        raise ValueError(
            "Expected 180 observations and 30 teams."
        )

    if not df["continuity"].between(0, 1).all():
        raise ValueError(
            "Continuity must fall between 0 and 1."
        )

    # Exposure should be fixed within each team.
    exposure_changes = (
        df.groupby("team_id")["exposure"].nunique()
    )

    if (exposure_changes != 1).any():
        raise ValueError(
            "Exposure changes within one or more teams."
        )

    # Centre exposure at its sample mean.
    # A one-unit increase in exposure10 represents
    # a 0.10 increase in payroll-to-cap ratio.
    mean_exposure = (
        df.drop_duplicates("team_id")["exposure"].mean()
    )

    df["exposure10"] = (
        df["exposure"] - mean_exposure
    ) / EXPOSURE_STEP

    # Create separate interaction variables for
    # every season except the 2022-23 reference.
    interaction_names = {}

    for season in SEASONS:
        if season == REFERENCE:
            continue

        variable = "exp_" + season.replace("-", "_")

        df[variable] = (
            df["exposure10"]
            * (df["season"] == season).astype(int)
        )

        interaction_names[season] = variable

    # Team fixed effects, season fixed effects,
    # and exposure-by-season interactions.
    formula = (
        "continuity ~ C(team_id) + C(season) + "
        + " + ".join(interaction_names.values())
    )

    model = smf.ols(
        formula=formula,
        data=df
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df["team_id"],
            "use_correction": True
        }
    )

    # Collect coefficients and clustered confidence
    # intervals. Convert proportions to percentage
    # points by multiplying by 100.
    ci = model.conf_int(alpha=0.05)
    rows = []

    for season in SEASONS:
        if season == REFERENCE:
            rows.append({
                "season": season,
                "coefficient_pp": 0.0,
                "se_pp": np.nan,
                "ci_lower_pp": np.nan,
                "ci_upper_pp": np.nan,
                "p_value": np.nan,
                "reference": True
            })
            continue

        variable = interaction_names[season]

        rows.append({
            "season": season,
            "coefficient_pp": model.params[variable] * 100,
            "se_pp": model.bse[variable] * 100,
            "ci_lower_pp": ci.loc[variable, 0] * 100,
            "ci_upper_pp": ci.loc[variable, 1] * 100,
            "p_value": model.pvalues[variable],
            "reference": False
        })

    results = pd.DataFrame(rows)

    RESULTS.parent.mkdir(
        parents=True, exist_ok=True
    )
    FIGURE.parent.mkdir(
        parents=True, exist_ok=True
    )

    results.to_csv(RESULTS, index=False)

    print("\nEVENT-STUDY REGRESSION")
    print("----------------------")
    print("Observations:", int(model.nobs))
    print("Teams:", df["team_id"].nunique())
    print("Reference season:", REFERENCE)
    print("Mean exposure:", round(mean_exposure, 4))
    print(
        "Interpretation: percentage-point change "
        "per 0.10 higher payroll-to-cap exposure."
    )

    print("\nEVENT-STUDY COEFFICIENTS")
    print(
        results.round(4).to_string(index=False)
    )

    # Joint test of all pre-policy interactions.
    # This is a diagnostic, not proof of parallel trends.
    pre_seasons = [
        "2019-20", "2020-21", "2021-22"
    ]

    restrictions = ", ".join(
        f"{interaction_names[s]} = 0"
        for s in pre_seasons
    )

    pre_test = model.wald_test(
        restrictions,
        scalar=True,
        use_f=False
    )

    print("\nJOINT PRE-POLICY COEFFICIENT TEST")
    print("Null: All three pre-policy coefficients = 0")
    print("Wald statistic:", round(float(pre_test.statistic), 4))
    print("P-value:", round(float(pre_test.pvalue), 4))
    print(
        "Note: Failing to reject does not establish "
        "parallel trends."
    )

    # Plot the dynamic exposure coefficients.
    fig, ax = plt.subplots(figsize=(10, 6))

    observed = results[
        ~results["reference"]
    ].copy()

    positions = {
        season: i for i, season in enumerate(SEASONS)
    }

    x = observed["season"].map(positions).to_numpy()
    y = observed["coefficient_pp"].to_numpy()

    lower = (
        y - observed["ci_lower_pp"].to_numpy()
    )
    upper = (
        observed["ci_upper_pp"].to_numpy() - y
    )

    ax.errorbar(
        x,
        y,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=5,
        linewidth=1.5,
        markersize=6,
        label="Estimated coefficient (95% CI)"
    )

    # Reference season is normalized to zero.
    ax.scatter(
        [positions[REFERENCE]],
        [0],
        marker="s",
        color="gray",
        zorder=4,
        label="Reference season"
    )

    ax.axhline(
        0, color="black", linewidth=1,
        linestyle="--"
    )

    # CBA introduced between 2022-23 and 2023-24.
    ax.axvline(
        3.5,
        color="gray",
        linestyle=":",
        linewidth=1.5
    )

    ax.text(
        3.58, 0.97,
        "2023 CBA",
        transform=ax.get_xaxis_transform(),
        va="top",
        fontsize=10
    )

    ax.set_xticks(range(len(SEASONS)))
    ax.set_xticklabels(SEASONS, rotation=30)

    ax.set_ylabel(
        "Continuity change (percentage points)\n"
        "per 0.10 higher pre-CBA exposure"
    )

    ax.set_xlabel("NBA Season")
    ax.set_title(
        "Roster Continuity: Continuous-Exposure Event Study"
    )

    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        FIGURE, dpi=300, bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved coefficients:", RESULTS)
    print("Saved figure:", FIGURE)
    print("\nDONE")


if __name__ == "__main__":
    main()
