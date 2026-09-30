# Dissertation Data Dictionary

## Research Question

How the NBA's second salary-cap apron reshaped demand for
cost-controlled labour and roster continuity.

## Sample Period

2018-19 through 2025-26, subject to consistent data availability.

---

# 1. Player-Team-Season Dataset

**Unit of observation:** Player × Team × Season

This dataset is used to test whether teams with greater pre-CBA
payroll exposure increased their utilization of cost-controlled labour
following the introduction of the second apron.

| Variable | Description | Type | Source | Status |
|---|---|---|---|---|
| player_id | Unique player identifier | String | TBD | Raw |
| player_name | Player name | String | TBD | Raw |
| team_id | Unique team identifier | String | TBD | Raw |
| season | NBA season | String | TBD | Raw |
| games | Games played for team | Integer | TBD | Raw |
| minutes | Minutes played for team | Float | TBD | Raw |
| salary | Player salary | Float | TBD | Raw |
| rookie_scale | Player is on rookie-scale contract | Binary | TBD | Constructed |
| minimum_contract | Player is on minimum contract | Binary | TBD | Constructed |
| cost_controlled | Player meets cost-controlled definition | Binary | TBD | Constructed |
| post | Post-second-apron period | Binary | N/A | Constructed |
| exposure | Team's pre-CBA payroll exposure | Float | Team dataset | Constructed |

---

# 2. Team-Season Dataset

**Unit of observation:** Team × Season

This dataset is used to test whether greater pre-CBA payroll exposure
reduced roster continuity following the introduction of the second apron.

| Variable | Description | Type | Source | Status |
|---|---|---|---|---|
| team_id | Unique team identifier | String | TBD | Raw |
| season | NBA season | String | TBD | Raw |
| payroll | Team payroll | Float | TBD | Raw |
| payroll_2022_23 | Team's predetermined 2022-23 payroll | Float | TBD | Constructed |
| exposure | Pre-CBA payroll exposure | Float | TBD | Constructed |
| post | Post-second-apron period | Binary | N/A | Constructed |
| total_minutes | Total team minutes | Float | Player dataset | Constructed |
| returning_minutes | Minutes played by returning players | Float | Player dataset | Constructed |
| continuity | Share of minutes played by returning players | Float | Player dataset | Constructed |

---

# Key Constructed Variables

## Pre-CBA Payroll Exposure

Treatment exposure is determined using team payroll in the 2022-23
season, prior to ratification of the 2023 CBA.

The baseline definition will be determined before the main analysis.
Alternative definitions may be used for robustness checks.

## Cost-Controlled Labour

The baseline definition will identify players on cost-controlled
contracts. Rookie-scale and minimum contracts will be retained as
separate indicators so that alternative definitions can be tested.

## Roster Continuity

Roster continuity is defined as:

Continuity = Returning Player Minutes / Total Team Minutes

A returning player is a player who was also on the same team's roster
during the previous season.

---

# Main Regression Variables

## Player-Level Analysis

Main interaction:

Rookie × Post × Exposure

Purpose: Test whether exposed teams differentially increased their
utilization of cost-controlled players following the policy.

## Team-Level Analysis

Main treatment interaction:

Post × Exposure

Main outcome:

Roster Continuity

Purpose: Test whether greater pre-CBA payroll exposure is associated
with a post-policy reduction in roster continuity.

---

# Data Collection Log

| Dataset | Seasons | Source | Collected | Cleaned |
|---|---|---|---|---|
| Player statistics | 2018-19 to 2025-26 | TBD | No | No |
| Player salaries | 2018-19 to 2025-26 | TBD | No | No |
| Player contracts | 2018-19 to 2025-26 | TBD | No | No |
| Team payroll | 2018-19 to 2025-26 | TBD | No | No |
| Rosters | 2018-19 to 2025-26 | TBD | No | No |
