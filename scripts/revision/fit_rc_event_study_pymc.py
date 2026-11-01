"""Random-coefficient event-study, PyMC NUTS fallback (Stan compile blocked on box).

Same model as rc_event_study.stan:
  shift_obs[i] = alpha + beta*shift_pred[i] + gamma*year_gap_z[i]
                + u_coach[i] + b_coach[i]*shift_pred[i] + epsilon[i]

Non-centered parameterization for coach random effects.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

DATA = Path(__file__).resolve().parents[2] / "data" / "nfl"
OUT = DATA / "sloan_revision_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def fit_one(label: str, paired_csv: Path, coach_col: str) -> dict:
    print(f"\n{'=' * 80}", flush=True)
    print(f"Fitting random-coefficient event-study for {label}", flush=True)
    print(f"{'=' * 80}", flush=True)

    df = pd.read_csv(paired_csv)
    df = df.dropna(subset=["shift_obs", "shift_pred", "prior_season", "new_season"]).copy()
    df["year_gap"] = df["new_season"].astype(int) - df["prior_season"].astype(int)
    yg_mean, yg_sd = df["year_gap"].mean(), df["year_gap"].std()
    if yg_sd == 0:
        yg_sd = 1
    df["year_gap_z"] = (df["year_gap"] - yg_mean) / yg_sd

    coaches = sorted(df[coach_col].unique())
    coach_idx = {c: i for i, c in enumerate(coaches)}
    df["coach_id"] = df[coach_col].map(coach_idx)
    K = len(coaches)
    N = len(df)
    print(f"  N events = {N}, K coaches = {K}, multi-event coaches = {(df[coach_col].value_counts() >= 2).sum()}", flush=True)

    # OLS reference
    X = np.column_stack([np.ones(N), df["shift_pred"].values])
    ols_coef, *_ = np.linalg.lstsq(X, df["shift_obs"].values, rcond=None)
    pearson_r = float(np.corrcoef(df["shift_pred"], df["shift_obs"])[0, 1])
    print(f"  OLS slope (paper): {ols_coef[1]:+.4f}", flush=True)
    print(f"  Pearson r: {pearson_r:+.4f}", flush=True)

    X2 = np.column_stack([np.ones(N), df["shift_pred"].values, df["year_gap_z"].values])
    ols2_coef, *_ = np.linalg.lstsq(X2, df["shift_obs"].values, rcond=None)
    print(f"  OLS + year_gap slope: {ols2_coef[1]:+.4f}", flush=True)

    coach_arr = df["coach_id"].values.astype(int)
    spred = df["shift_pred"].values.astype(float)
    sobs = df["shift_obs"].values.astype(float)
    ygap = df["year_gap_z"].values.astype(float)

    print(f"\n  Building PyMC model...", flush=True)
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=0.05)
        beta = pm.Normal("beta", mu=0.35, sigma=0.5)
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.02)

        sigma_u = pm.HalfNormal("sigma_u", sigma=0.05)
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.1)

        u_raw = pm.Normal("u_coach_raw", mu=0.0, sigma=1.0, shape=K)
        b_raw = pm.Normal("b_coach_raw", mu=0.0, sigma=1.0, shape=K)
        u_coach = pm.Deterministic("u_coach", u_raw * sigma_u)
        b_coach = pm.Deterministic("b_coach", b_raw * sigma_b)

        mu = (
            alpha
            + beta * spred
            + gamma * ygap
            + u_coach[coach_arr]
            + b_coach[coach_arr] * spred
        )
        pm.Normal("y", mu=mu, sigma=sigma, observed=sobs)

        # Derived per-coach effective slopes
        coach_slope = pm.Deterministic("coach_slope", beta + b_coach)

        print(f"  Sampling (4 chains x 3000 iter, tune=2000, target_accept=0.995)...", flush=True)
        idata = pm.sample(
            3000,
            tune=2000,
            chains=4,
            cores=1,
            target_accept=0.995,
            max_treedepth=15,
            progressbar=False,
            random_seed=42,
        )

    post = idata.posterior
    alpha_p = post["alpha"].values.flatten()
    beta_p = post["beta"].values.flatten()
    gamma_p = post["gamma"].values.flatten()
    sigma_u_p = post["sigma_u"].values.flatten()
    sigma_b_p = post["sigma_b"].values.flatten()
    sigma_p = post["sigma"].values.flatten()

    def summ(s, name):
        return {
            "name": name,
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "sd": float(s.std()),
            "q025": float(np.percentile(s, 2.5)),
            "q975": float(np.percentile(s, 97.5)),
            "P(>0)": float((s > 0).mean()),
        }

    summaries = [
        summ(alpha_p, "alpha"),
        summ(beta_p, "beta_population"),
        summ(gamma_p, "gamma_year_gap"),
        summ(sigma_u_p, "sigma_u_coach_intercept"),
        summ(sigma_b_p, "sigma_b_coach_slope"),
        summ(sigma_p, "sigma_residual"),
    ]
    print(f"\n  Posterior summaries:", flush=True)
    for s in summaries:
        print(f"    {s['name']:32s}  mean={s['mean']:+.4f}  [95%: {s['q025']:+.4f}, {s['q975']:+.4f}]  P(>0)={s['P(>0)']:.3f}", flush=True)

    print(f"\n  OLS slope vs Bayesian shrunk slope:", flush=True)
    print(f"    OLS              = {ols_coef[1]:+.4f}", flush=True)
    print(f"    OLS + year_gap   = {ols2_coef[1]:+.4f}", flush=True)
    print(f"    Bayes (shrunk)   = {beta_p.mean():+.4f}  [95%: {np.percentile(beta_p, 2.5):+.4f}, {np.percentile(beta_p, 97.5):+.4f}]", flush=True)
    print(f"    shrinkage delta  = {beta_p.mean() - ols_coef[1]:+.4f}", flush=True)

    # Per-coach effective slopes
    coach_slope_samples = post["coach_slope"].values.reshape(-1, K)
    coach_intercept_samples = (post["alpha"].values.reshape(-1, 1) + post["u_coach"].values.reshape(-1, K))
    coach_df = pd.DataFrame({
        "coach": coaches,
        "n_events": [df[df[coach_col] == c].shape[0] for c in coaches],
        "slope_posterior_mean": coach_slope_samples.mean(axis=0),
        "slope_posterior_q025": np.percentile(coach_slope_samples, 2.5, axis=0),
        "slope_posterior_q975": np.percentile(coach_slope_samples, 97.5, axis=0),
        "intercept_posterior_mean": coach_intercept_samples.mean(axis=0),
    }).sort_values("slope_posterior_mean", ascending=False)
    coach_df.to_csv(OUT / f"{label}_coach_effects.csv", index=False)
    print(f"\n  Saved per-coach effects -> {OUT / f'{label}_coach_effects.csv'}", flush=True)

    print(f"\n  TOP-5 coaches by posterior mean slope:", flush=True)
    for _, row in coach_df.head(5).iterrows():
        print(f"    {row['coach']:30s}  slope = {row['slope_posterior_mean']:+.3f}  [{row['slope_posterior_q025']:+.3f}, {row['slope_posterior_q975']:+.3f}]  n={int(row['n_events'])}", flush=True)
    print(f"\n  BOTTOM-5 coaches by posterior mean slope:", flush=True)
    for _, row in coach_df.tail(5).iterrows():
        print(f"    {row['coach']:30s}  slope = {row['slope_posterior_mean']:+.3f}  [{row['slope_posterior_q025']:+.3f}, {row['slope_posterior_q975']:+.3f}]  n={int(row['n_events'])}", flush=True)

    result = {
        "label": label,
        "N_events": int(N),
        "K_coaches": int(K),
        "multi_event_coaches": int((df[coach_col].value_counts() >= 2).sum()),
        "OLS_slope": float(ols_coef[1]),
        "OLS_slope_plus_year_gap": float(ols2_coef[1]),
        "Bayes_slope_mean": float(beta_p.mean()),
        "Bayes_slope_median": float(np.median(beta_p)),
        "Bayes_slope_q025": float(np.percentile(beta_p, 2.5)),
        "Bayes_slope_q975": float(np.percentile(beta_p, 97.5)),
        "shrinkage_delta": float(beta_p.mean() - ols_coef[1]),
        "year_gap_mean": float(yg_mean),
        "year_gap_sd": float(yg_sd),
        "posterior_summaries": summaries,
        "pearson_r": pearson_r,
    }
    with open(OUT / f"{label}_rc_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame({
        "alpha": alpha_p,
        "beta": beta_p,
        "gamma": gamma_p,
        "sigma_u": sigma_u_p,
        "sigma_b": sigma_b_p,
        "sigma": sigma_p,
    }).to_parquet(OUT / f"{label}_rc_posterior.parquet")
    print(f"  Saved -> {OUT / f'{label}_rc_summary.json'} + {OUT / f'{label}_rc_posterior.parquet'}", flush=True)
    return result


def main():
    print("=" * 80, flush=True)
    print("SLOAN REVISION #1: Random-coefficient event-study (PyMC NUTS)", flush=True)
    print("=" * 80, flush=True)

    hc_result = fit_one("hc", DATA / "hc_predictive_validation_paired.csv", "HC")
    oc_result = fit_one("oc", DATA / "oc_predictive_validation_paired.csv", "OC")

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: OLS vs Bayesian shrunk slopes", flush=True)
    print("=" * 80, flush=True)
    print(f"  HC: OLS {hc_result['OLS_slope']:+.3f} -> Bayes {hc_result['Bayes_slope_mean']:+.3f}  [95%: {hc_result['Bayes_slope_q025']:+.3f}, {hc_result['Bayes_slope_q975']:+.3f}]  n={hc_result['N_events']}", flush=True)
    print(f"  OC: OLS {oc_result['OLS_slope']:+.3f} -> Bayes {oc_result['Bayes_slope_mean']:+.3f}  [95%: {oc_result['Bayes_slope_q025']:+.3f}, {oc_result['Bayes_slope_q975']:+.3f}]  n={oc_result['N_events']}", flush=True)
    print(f"\n  Paper claim: HC slope 0.31 (contaminated, 13th pct placebo); OC slope 0.45 (clean, 100th pct).", flush=True)


if __name__ == "__main__":
    main()
