"""Revision #1 follow-ups: locate Bayes shrunk slope in OLS-placebo distribution,
test year_gap covariate effect on placebo, refit Bayes on propensity-matched events,
build posterior-band figure.

Outputs:
  data/nfl/sloan_revision_outputs/placebo_comparison.json
  data/nfl/sloan_revision_outputs/propensity_refit.json
  data/nfl/sloan_revision_outputs/predictive_validation_with_posterior.png
"""
from __future__ import annotations
import json
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")

# RTools path for completeness
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nfl"
OUT = DATA / "sloan_revision_outputs"


def load_paired(label: str, coach_col: str):
    df = pd.read_csv(DATA / f"{label}_predictive_validation_paired.csv")
    df = df.dropna(subset=["shift_obs", "shift_pred", "prior_season", "new_season"]).copy()
    df["year_gap"] = df["new_season"].astype(int) - df["prior_season"].astype(int)
    yg_mean, yg_sd = df["year_gap"].mean(), df["year_gap"].std() or 1
    df["year_gap_z"] = (df["year_gap"] - yg_mean) / yg_sd
    coaches = sorted(df[coach_col].unique())
    coach_idx = {c: i for i, c in enumerate(coaches)}
    df["coach_id"] = df[coach_col].map(coach_idx)
    return df, coaches


def fit_rc(df, coaches, sigma_b_prior=0.5, draws=2000, tune=1500):
    """Quick PyMC NUTS hierarchical fit. Returns population beta posterior + per-coach posteriors."""
    K = len(coaches)
    coach_arr = df["coach_id"].values.astype(int)
    spred = df["shift_pred"].values.astype(float)
    sobs = df["shift_obs"].values.astype(float)
    ygap = df["year_gap_z"].values.astype(float)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=0.05)
        beta = pm.Normal("beta", mu=0.35, sigma=0.5)
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.02)
        sigma_u = pm.HalfNormal("sigma_u", sigma=0.05)
        sigma_b = pm.HalfNormal("sigma_b", sigma=sigma_b_prior)
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        u_raw = pm.Normal("u_coach_raw", mu=0.0, sigma=1.0, shape=K)
        b_raw = pm.Normal("b_coach_raw", mu=0.0, sigma=1.0, shape=K)
        u_coach = u_raw * sigma_u
        b_coach = b_raw * sigma_b
        mu = alpha + beta * spred + gamma * ygap + u_coach[coach_arr] + b_coach[coach_arr] * spred
        pm.Normal("y", mu=mu, sigma=sigma, observed=sobs)
        coach_slope = pm.Deterministic("coach_slope", beta + b_coach)
        idata = pm.sample(draws, tune=tune, chains=4, cores=1, target_accept=0.995,
                          max_treedepth=15, progressbar=False, random_seed=42)
    return idata


def ols_slope(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


def main():
    print("=" * 80, flush=True)
    print("Revision #1 follow-ups (placebo + propensity + figure)", flush=True)
    print("=" * 80, flush=True)

    # === 1. Locate Bayes slope in OLS-placebo distribution ===
    robust = pd.read_csv(DATA / "predictive_validation_robustness.csv")
    rd = dict(zip(robust["metric"], robust["value"]))
    print(f"\nExisting OLS-placebo results:", flush=True)
    print(f"  HC: observed={rd['hc_observed_slope']:.4f}  placebo_med={rd['hc_placebo_median_slope']:.4f}  CI[{rd['hc_placebo_ci_lo']:.4f},{rd['hc_placebo_ci_hi']:.4f}]  obs@{rd['hc_observed_pct']:.1f}th pct", flush=True)
    print(f"  OC: observed={rd['oc_observed_slope']:.4f}  placebo_med={rd['oc_placebo_median_slope']:.4f}  CI[{rd['oc_placebo_ci_lo']:.4f},{rd['oc_placebo_ci_hi']:.4f}]  obs@{rd['oc_observed_pct']:.1f}th pct", flush=True)

    hc_summary = json.load(open(OUT / "hc_rc_summary.json"))
    oc_summary = json.load(open(OUT / "oc_rc_summary.json"))

    # Where does Bayes shrunk fall in OLS placebo distribution (approximate)?
    # Assume placebo distribution is roughly normal with given mean/CI
    def placebo_z_and_pct(observed, med, ci_lo, ci_hi):
        sd = (ci_hi - ci_lo) / (2 * 1.96)
        z = (observed - med) / sd if sd > 0 else 0
        from scipy.stats import norm
        pct = norm.cdf(z) * 100
        return z, pct

    hc_bayes = hc_summary["Bayes_slope_mean"]
    oc_bayes = oc_summary["Bayes_slope_mean"]
    hc_z, hc_pct = placebo_z_and_pct(hc_bayes, rd["hc_placebo_median_slope"], rd["hc_placebo_ci_lo"], rd["hc_placebo_ci_hi"])
    oc_z, oc_pct = placebo_z_and_pct(oc_bayes, rd["oc_placebo_median_slope"], rd["oc_placebo_ci_lo"], rd["oc_placebo_ci_hi"])
    print(f"\nBayes shrunk slope in OLS-placebo distribution:", flush=True)
    print(f"  HC: Bayes {hc_bayes:.4f}  -> z={hc_z:+.2f}  pct={hc_pct:.1f}th  (OLS was at {rd['hc_observed_pct']:.1f}th)", flush=True)
    print(f"  OC: Bayes {oc_bayes:.4f}  -> z={oc_z:+.2f}  pct={oc_pct:.1f}th  (OLS was at {rd['oc_observed_pct']:.1f}th)", flush=True)

    # === 2. Refit Bayes on propensity-matched events ===
    print("\n" + "=" * 80, flush=True)
    print("PROPENSITY REFIT", flush=True)
    print("=" * 80, flush=True)
    # Load propensity-matched results (paper's matched events)
    hc_prop = pd.read_csv(DATA / "hc_propensity_matched_results.csv")
    print(f"  hc_propensity_matched_results columns: {list(hc_prop.columns)}", flush=True)
    print(f"  rows: {len(hc_prop)}", flush=True)
    print(hc_prop.head().to_string(), flush=True)

    # === 3. Build figure with posterior bands ===
    print("\n" + "=" * 80, flush=True)
    print("FIGURE: predictive_validation with Bayesian posterior bands", flush=True)
    print("=" * 80, flush=True)

    hc_df = pd.read_csv(DATA / "hc_predictive_validation_paired.csv").dropna(subset=["shift_obs", "shift_pred"])
    oc_df = pd.read_csv(DATA / "oc_predictive_validation_paired.csv").dropna(subset=["shift_obs", "shift_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, df, label, summary, coach_col in [
        (axes[0], hc_df, "HC pass-rate portability", hc_summary, "HC"),
        (axes[1], oc_df, "OC RZ-shotgun portability", oc_summary, "OC"),
    ]:
        x = df["shift_pred"].values
        y = df["shift_obs"].values
        ax.scatter(x, y, s=18, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.4, zorder=3)
        # 45-degree line
        lo, hi = -0.4, 0.4
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, label="Perfect transfer", zorder=1)
        # OLS line
        ols = np.polyfit(x, y, 1)
        xx = np.linspace(lo, hi, 100)
        ax.plot(xx, ols[1] + ols[0] * xx, "-", color="orange", linewidth=2, label=f"OLS slope = {ols[0]:+.3f}", zorder=2)
        # Bayes line + 95% band
        beta_p = summary["Bayes_slope_mean"]
        beta_lo, beta_hi = summary["Bayes_slope_q025"], summary["Bayes_slope_q975"]
        alpha_p = summary["posterior_summaries"][0]["mean"]
        ax.plot(xx, alpha_p + beta_p * xx, "-", color="#d62728", linewidth=2.5,
                label=f"Bayes shrunk = {beta_p:+.3f}\n[95% CI: {beta_lo:+.3f}, {beta_hi:+.3f}]", zorder=4)
        # Band: simulate at lo,hi range
        ax.fill_between(xx, alpha_p + beta_lo * xx, alpha_p + beta_hi * xx, color="#d62728", alpha=0.15, zorder=1)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Predicted shift\n(coach prior − new-team pre)", fontsize=10)
        ax.set_ylabel("Observed shift\n(new-team post − new-team pre)", fontsize=10)
        ax.set_title(f"{label}\nn = {summary['N_events']}, K coaches = {summary['K_coaches']}", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Predictive validation: OLS vs Bayesian shrunk slopes\n(Sloan revision #1 — random-coefficient event-study with coach-level shrinkage)", fontsize=11)
    fig_path = OUT / "predictive_validation_with_posterior.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {fig_path}", flush=True)
    plt.close(fig)

    # === Save consolidated summary ===
    consolidated = {
        "placebo_comparison": {
            "hc": {
                "ols_observed": float(rd["hc_observed_slope"]),
                "ols_observed_pct": float(rd["hc_observed_pct"]),
                "bayes_shrunk_slope": float(hc_bayes),
                "bayes_in_ols_placebo_pct": float(hc_pct),
                "bayes_in_ols_placebo_z": float(hc_z),
                "placebo_median": float(rd["hc_placebo_median_slope"]),
                "placebo_95ci": [float(rd["hc_placebo_ci_lo"]), float(rd["hc_placebo_ci_hi"])],
            },
            "oc": {
                "ols_observed": float(rd["oc_observed_slope"]),
                "ols_observed_pct": float(rd["oc_observed_pct"]),
                "bayes_shrunk_slope": float(oc_bayes),
                "bayes_in_ols_placebo_pct": float(oc_pct),
                "bayes_in_ols_placebo_z": float(oc_z),
                "placebo_median": float(rd["oc_placebo_median_slope"]),
                "placebo_95ci": [float(rd["oc_placebo_ci_lo"]), float(rd["oc_placebo_ci_hi"])],
            },
        },
    }
    with open(OUT / "placebo_comparison.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\n  Saved consolidated summary -> {OUT / 'placebo_comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
