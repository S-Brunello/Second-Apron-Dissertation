
"""Dynamic exposure models using team-game continuity.

Three time resolutions:
1. Calendar month
2. NBA-season-relative three-part periods
3. NBA season

All models use the same team-game data and cluster by team.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/team_game_continuity.csv"
OUTPUT = ROOT / "output/tables"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE, dtype={"game_id": str})
df["game_date"] = pd.to_datetime(df["game_date"])

assert len(df) == 16398
assert df["team_id"].nunique() == 30
assert not df.duplicated(["team_id", "game_id"]).any()
assert df[["continuity_pct", "exposure"]].notna().all().all()

# A one-unit change in exposure_scaled is a 0.10
# increase in payroll/cap exposure.
df["exposure_scaled"] = (
    df["exposure"] - df["exposure"].mean()
) / 0.10

# Season-relative time periods

month_order = (
    df[["nba_season", "calendar_month"]]
    .drop_duplicates()
    .sort_values(["nba_season", "calendar_month"])
)

month_order["month_index"] = (
    month_order.groupby("nba_season").cumcount()
)

month_order["n_months"] = (
    month_order.groupby("nba_season")["calendar_month"]
    .transform("size")
)

month_order["block"] = (
    np.floor(
        3 * month_order["month_index"]
        / month_order["n_months"]
    ).astype(int) + 1
)

month_order["season_period"] = (
    month_order["nba_season"]
    + " Q"
    + month_order["block"].astype(str)
)

df = df.merge(
    month_order[
        ["nba_season", "calendar_month", "season_period"]
    ],
    on=["nba_season", "calendar_month"],
    how="left",
    validate="many_to_one"
)


# Regression function

def estimate_model(data, time_column, reference, label):
    data = data.copy()
    data["period"] = data[time_column].astype(str)

    periods = (
        data.groupby("period")["game_date"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

    if reference not in periods:
        raise ValueError(f"Reference period not found: {reference}")

    terms = []
    term_to_period = {}

    for i, period in enumerate(periods):
        if period == reference:
            continue

        term = f"exposure_x_{i:02d}"
        data[term] = (
            data["exposure_scaled"]
            * (data["period"] == period).astype(int)
        )

        terms.append(term)
        term_to_period[term] = period

    formula = (
        "continuity_pct ~ C(team_id) + C(period) + "
        + " + ".join(terms)
    )

    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["team_id"]},
        use_t=True
    )

    confidence = model.conf_int()
    
    # Joint test of pre-policy exposure interactions.
    # All game periods before October 2023 are pre-policy,
    # including April 2023.

    pre_periods = {
        period for period in periods
        if data.loc[
            data["period"] == period, "game_date"
        ].min() < pd.Timestamp("2023-07-01")
    }

    pre_terms = [
        term for term in terms
        if term_to_period[term] in pre_periods
    ]

    if pre_terms:
        restrictions = ", ".join(
            f"{term} = 0" for term in pre_terms
        )

        pre_test = model.f_test(restrictions)

        print("\nJOINT PRE-POLICY TEST")
        print("Restrictions:", len(pre_terms))
        print("F-statistic:", round(float(pre_test.fvalue), 4))
        print("p-value:", round(float(pre_test.pvalue), 4))


    results = []

    for term in terms:
        results.append({
            "period": term_to_period[term],
            "reference": reference,
            "coefficient_pp": model.params[term],
            "std_error": model.bse[term],
            "p_value": model.pvalues[term],
            "ci_lower": confidence.loc[term, 0],
            "ci_upper": confidence.loc[term, 1]
        })

    results = pd.DataFrame(results)

    filename = OUTPUT / f"dynamic_game_{label}.csv"
    results.to_csv(filename, index=False)

    print(f"\n{label.upper()} MODEL")
    print("Observations:", int(model.nobs))
    print("Teams:", data["team_id"].nunique())
    print("Time periods:", len(periods))
    print("Reference:", reference)
    print("R-squared:", round(model.rsquared, 4))
    print("Exposure interaction coefficients:", len(terms))

    print("\nMOST RECENT COEFFICIENTS")
    print(
        results.tail(8).round(4).to_string(index=False)
    )
    print("\nSaved:", filename)

    return model, results


# Estimate at three time resolutions

monthly_model, monthly_results = estimate_model(
    df,
    "calendar_month",
    "2023-03",
    "monthly"
)

quarterly_model, quarterly_results = estimate_model(
    df,
    "season_period",
    "2022-23 Q3",
    "quarterly"
)

annual_model, annual_results = estimate_model(
    df,
    "nba_season",
    "2022-23",
    "annual"
)

print("\nALL THREE MODELS ESTIMATED.")
print("Existing datasets and annual regressions unchanged.")
