"""Propensity-controlled refit of random-coefficient event-study.

Method: regression adjustment for team-quality covariates in the hierarchical model
(equivalent to propensity matching for slope-tests, more efficient).

Covariates added to RC model:
  - prior_team_wins_prev  : wins of prior team in season prior_season-1 (z-scored)
  - new_team_wins_prev    : wins of new team in season new_season-1 (z-scored)
  - hc_tenure_years_prior : years coach was at prior team (z-scored)

Test: does Bayes beta_shrunk remain stable / significant after these adjustments?
"""
from __future__ import annotations
import json
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")

for rtools in ["C:/rtools44", "C:/rtools43", "C:/rtools42", "C:/rtools40"]:
    if os.path.exists(rtools):
        for sub in ["mingw64/bin", "usr/bin"]:
            p = f"{rtools}/{sub}"
            if os.path.exists(p) and p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")
        break

import numpy as np
import pandas as pd
import pymc as pm

DATA = Path(__file__).resolve().parents[2] / "data" / "nfl"
OUT = DATA / "sloan_revision_outputs"


def team_wins_by_season() -> pd.DataFrame:
    """Compute team wins per (team, season) from games panel."""
    games = pd.read_parquet(DATA / "games" / "games.parquet")
    games = games[games["game_type"] == "REG"].copy()
    home = games[["season", "home_team", "home_score", "away_score"]].copy()
    home["team"] = home["home_team"]; home["pf"] = home["home_score"]; home["pa"] = home["away_score"]
    away = games[["season", "away_team", "home_score", "away_score"]].copy()
    away["team"] = away["away_team"]; away["pf"] = away["away_score"]; away["pa"] = away["home_score"]
    long = pd.concat([home[["season", "team", "pf", "pa"]], away[["season", "team", "pf", "pa"]]], ignore_index=True)
    long = long.dropna(subset=["pf", "pa"])
    long["win"] = (long["pf"] > long["pa"]).astype(int)
    long["tie"] = (long["pf"] == long["pa"]).astype(int)
    agg = long.groupby(["team", "season"]).agg(
        wins=("win", "sum"), ties=("tie", "sum"),
        pf=("pf", "sum"), pa=("pa", "sum"), games=("win", "size"),
    ).reset_index()
    agg["w_eq"] = agg["wins"] + 0.5 * agg["ties"]  # weighted wins
    return agg


def tenure_at_team(coach_seasons: pd.DataFrame, coach: str, team: str, season_end: int) -> int:
    """How many consecutive seasons (ending at season_end) was coach at team?"""
    cs = coach_seasons[(coach_seasons["coach_name"] == coach) & (coach_seasons["team"] == team)].copy()
    cs = cs[cs["season"] <= season_end].sort_values("season", ascending=False)
    if cs.empty:
        return 0
    tenure = 0
    expected = season_end
    for _, row in cs.iterrows():
        if int(row["season"]) == expected:
            tenure += 1
            expected -= 1
        else:
            break
    return tenure


def attach_covariates(df: pd.DataFrame, coach_col: str, role_str: str) -> pd.DataFrame:
    """Add prior_team_wins_prev, new_team_wins_prev, hc_tenure_prior to each event."""
    wins = team_wins_by_season()
    cs = pd.read_parquet(DATA / "coaching" / "coach_seasons.parquet")
    cs = cs[cs["role"].isin(["head_coach", "offensive_coordinator", "defensive_coordinator"])].copy()

    out = df.copy()
    out["prior_team_wins_prev"] = np.nan
    out["new_team_wins_prev"] = np.nan
    out["tenure_years_prior"] = np.nan

    for i, row in out.iterrows():
        coach = row[coach_col]
        prior_team = row["prior_team"]
        prior_season = int(row["prior_season"])
        new_team = row["new_team"]
        new_season = int(row["new_season"])

        # prior_team wins in prior_season - 1
        pw = wins[(wins["team"] == prior_team) & (wins["season"] == prior_season - 1)]
        if len(pw):
            out.at[i, "prior_team_wins_prev"] = float(pw["w_eq"].iloc[0])
        # new_team wins in new_season - 1
        nw = wins[(wins["team"] == new_team) & (wins["season"] == new_season - 1)]
        if len(nw):
            out.at[i, "new_team_wins_prev"] = float(nw["w_eq"].iloc[0])
        # tenure at prior team ending at prior_season
        out.at[i, "tenure_years_prior"] = float(tenure_at_team(cs, coach, prior_team, prior_season))

    return out


def fit_with_covariates(label: str, paired_csv: str, coach_col: str):
    print(f"\n{'=' * 80}", flush=True)
    print(f"Propensity-controlled re-fit: {label}", flush=True)
    print(f"{'=' * 80}", flush=True)

    df = pd.read_csv(DATA / paired_csv)
    df = df.dropna(subset=["shift_obs", "shift_pred", "prior_season", "new_season"]).copy()
    df["year_gap"] = df["new_season"].astype(int) - df["prior_season"].astype(int)
    df["year_gap_z"] = (df["year_gap"] - df["year_gap"].mean()) / (df["year_gap"].std() or 1)

    df = attach_covariates(df, coach_col, label)
    cov_cols = ["prior_team_wins_prev", "new_team_wins_prev", "tenure_years_prior"]
    n_before = len(df)
    df = df.dropna(subset=cov_cols).copy()
    print(f"  events: {len(df)} (dropped {n_before - len(df)} with missing covariates)", flush=True)

    # Z-score covariates
    for c in cov_cols:
        df[c + "_z"] = (df[c] - df[c].mean()) / (df[c].std() or 1)

    # Encode coach
    coaches = sorted(df[coach_col].unique())
    coach_idx = {c: i for i, c in enumerate(coaches)}
    df["coach_id"] = df[coach_col].map(coach_idx)
    K = len(coaches)
    N = len(df)

    print(f"  N events = {N}, K coaches = {K}, multi-event = {(df[coach_col].value_counts() >= 2).sum()}", flush=True)
    print(f"  covariate mean / SD:", flush=True)
    for c in cov_cols:
        print(f"    {c}: {df[c].mean():.2f} ± {df[c].std():.2f}", flush=True)

    # OLS reference (no covariates, then with covariates)
    X0 = np.column_stack([np.ones(N), df["shift_pred"].values])
    ols_no, *_ = np.linalg.lstsq(X0, df["shift_obs"].values, rcond=None)
    print(f"\n  OLS slope (no covariates): {ols_no[1]:+.4f}", flush=True)

    X1 = np.column_stack([
        np.ones(N), df["shift_pred"].values,
        df["year_gap_z"].values,
        df["prior_team_wins_prev_z"].values,
        df["new_team_wins_prev_z"].values,
        df["tenure_years_prior_z"].values,
    ])
    ols_with, *_ = np.linalg.lstsq(X1, df["shift_obs"].values, rcond=None)
    print(f"  OLS slope (+ covariates): {ols_with[1]:+.4f}", flush=True)
    print(f"    year_gap: {ols_with[2]:+.4f}", flush=True)
    print(f"    prior_team_wins: {ols_with[3]:+.4f}", flush=True)
    print(f"    new_team_wins:   {ols_with[4]:+.4f}", flush=True)
    print(f"    tenure:          {ols_with[5]:+.4f}", flush=True)

    # Bayesian RC model with covariates
    coach_arr = df["coach_id"].values.astype(int)
    spred = df["shift_pred"].values.astype(float)
    sobs = df["shift_obs"].values.astype(float)
    ygap = df["year_gap_z"].values.astype(float)
    ptw = df["prior_team_wins_prev_z"].values.astype(float)
    ntw = df["new_team_wins_prev_z"].values.astype(float)
    ten = df["tenure_years_prior_z"].values.astype(float)

    print(f"\n  Fitting hierarchical model with covariates...", flush=True)
    with pm.Model():
        alpha = pm.Normal("alpha", mu=0.0, sigma=0.05)
        beta = pm.Normal("beta", mu=0.35, sigma=0.5)
        gamma_yg = pm.Normal("gamma_yg", mu=0.0, sigma=0.02)
        c_ptw = pm.Normal("c_ptw", mu=0.0, sigma=0.05)
        c_ntw = pm.Normal("c_ntw", mu=0.0, sigma=0.05)
        c_ten = pm.Normal("c_ten", mu=0.0, sigma=0.05)
        sigma_u = pm.HalfNormal("sigma_u", sigma=0.05)
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        u_raw = pm.Normal("u_raw", mu=0.0, sigma=1.0, shape=K)
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, shape=K)
        u_coach = u_raw * sigma_u
        b_coach = b_raw * sigma_b
        mu = (alpha
              + beta * spred
              + gamma_yg * ygap
              + c_ptw * ptw
              + c_ntw * ntw
              + c_ten * ten
              + u_coach[coach_arr]
              + b_coach[coach_arr] * spred)
        pm.Normal("y", mu=mu, sigma=sigma, observed=sobs)
        idata = pm.sample(3000, tune=2000, chains=4, cores=1, target_accept=0.995,
                          max_treedepth=15, progressbar=False, random_seed=42)

    post = idata.posterior
    beta_p = post["beta"].values.flatten()
    print(f"\n  Bayes slope w/ covariates: {beta_p.mean():+.4f}  [95%: {np.percentile(beta_p, 2.5):+.4f}, {np.percentile(beta_p, 97.5):+.4f}]  P(>0)={(beta_p > 0).mean():.3f}", flush=True)

    cov_betas = {}
    for name in ["c_ptw", "c_ntw", "c_ten", "gamma_yg"]:
        s = post[name].values.flatten()
        cov_betas[name] = {
            "mean": float(s.mean()),
            "q025": float(np.percentile(s, 2.5)),
            "q975": float(np.percentile(s, 97.5)),
            "P(>0)": float((s > 0).mean()),
        }
        print(f"    {name}: mean={s.mean():+.4f}  [{np.percentile(s, 2.5):+.4f}, {np.percentile(s, 97.5):+.4f}]  P(>0)={(s > 0).mean():.3f}", flush=True)

    # Comparison to no-covariate result
    summary = json.load(open(OUT / f"{label}_rc_summary.json"))
    print(f"\n  COMPARISON (Bayes beta):", flush=True)
    print(f"    no covariates:  {summary['Bayes_slope_mean']:+.4f}  [{summary['Bayes_slope_q025']:+.4f}, {summary['Bayes_slope_q975']:+.4f}]", flush=True)
    print(f"    w/ covariates:  {beta_p.mean():+.4f}  [{np.percentile(beta_p, 2.5):+.4f}, {np.percentile(beta_p, 97.5):+.4f}]", flush=True)
    print(f"    delta:          {beta_p.mean() - summary['Bayes_slope_mean']:+.4f}", flush=True)

    result = {
        "label": label,
        "N_events_with_covariates": int(N),
        "K_coaches": int(K),
        "OLS_slope_no_cov": float(ols_no[1]),
        "OLS_slope_with_cov": float(ols_with[1]),
        "Bayes_slope_no_cov_mean": float(summary["Bayes_slope_mean"]),
        "Bayes_slope_no_cov_q025": float(summary["Bayes_slope_q025"]),
        "Bayes_slope_no_cov_q975": float(summary["Bayes_slope_q975"]),
        "Bayes_slope_with_cov_mean": float(beta_p.mean()),
        "Bayes_slope_with_cov_q025": float(np.percentile(beta_p, 2.5)),
        "Bayes_slope_with_cov_q975": float(np.percentile(beta_p, 97.5)),
        "delta": float(beta_p.mean() - summary["Bayes_slope_mean"]),
        "covariate_betas": cov_betas,
    }
    with open(OUT / f"{label}_propensity_refit.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved -> {OUT / f'{label}_propensity_refit.json'}", flush=True)
    return result


def main():
    print("=" * 80, flush=True)
    print("Sloan Revision #1c — propensity-controlled Bayesian refit", flush=True)
    print("=" * 80, flush=True)

    hc_result = fit_with_covariates("hc", "hc_predictive_validation_paired.csv", "HC")
    oc_result = fit_with_covariates("oc", "oc_predictive_validation_paired.csv", "OC")

    print("\n" + "=" * 80, flush=True)
    print("ROBUSTNESS SUMMARY: Bayes beta under propensity-adjustment", flush=True)
    print("=" * 80, flush=True)
    for label, r in [("HC", hc_result), ("OC", oc_result)]:
        print(f"  {label}: no_cov {r['Bayes_slope_no_cov_mean']:+.3f}  w/ cov {r['Bayes_slope_with_cov_mean']:+.3f}  delta {r['delta']:+.3f}", flush=True)
        print(f"        95% CI no_cov: [{r['Bayes_slope_no_cov_q025']:+.3f}, {r['Bayes_slope_no_cov_q975']:+.3f}]", flush=True)
        print(f"        95% CI w/ cov: [{r['Bayes_slope_with_cov_q025']:+.3f}, {r['Bayes_slope_with_cov_q975']:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
