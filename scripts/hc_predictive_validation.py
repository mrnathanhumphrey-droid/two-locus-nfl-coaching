"""HC predictive validation — does incoming HC's prior pass-rate profile
forecast the new team's pass-rate shift at the HC-change event?

The Layer 2a finding (HC change → pass-rate shift 1.36x p=0.002) tells us
the latent exists. This test asks whether it's FORECASTABLE from individual
HC's prior data, or whether the latent is team-adapted (HC + roster co-form
the new profile, and you can't predict it from HC alone).

Design (retread HCs only — HCs with prior HC tenure in window):
  For each event where HC_X moves from team A to team B at offseason S:
    profile_HC_X_at_A: pass_rate_residual mean over HC_X's last K games at A
    profile_B_under_prior: pass_rate_residual mean over team B's last K
                            games of season S-1 (under prior HC at B)
    profile_B_under_X: pass_rate_residual mean over team B's first K games
                       of season S (under HC_X)

  Observed shift:   shift_obs = profile_B_under_X - profile_B_under_prior
  Predicted shift:  shift_pred = profile_HC_X_at_A - profile_B_under_prior
                    (i.e., "the new HC should pull team B toward their own
                     prior profile")

  Test: corr(shift_obs, shift_pred) across retread events.
  Also: OLS regression shift_obs ~ shift_pred, report beta (calibration)
        and R^2.
  Null: permutation test on shift_pred labels.

If r > 0 with p < 0.05 -> HC profile is PORTABLE, latent is forecastable.
If r ≈ 0 -> HC + roster co-form new profile, latent is TEAM-ADAPTED.

This is the falsification gate for the forecastability claim.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_change_event_study import (
    PFR_TO_NFL, LEGACY_REMAP, build_staff_panel, N_PERM,
)

ROOT = Path(__file__).resolve().parent.parent / "data"
K = 8
N_PERM_TEST = 5000


def load_offensive_residuals():
    sr = pd.read_parquet(ROOT / "games/state_residuals.parquet")
    if "week" not in sr.columns:
        games = pd.read_parquet(ROOT / "games/games.parquet")
        sr = sr.merge(games[["game_id","week"]], on="game_id", how="left")
    sr["team_canon"] = sr["posteam"].map(LEGACY_REMAP).fillna(sr["posteam"])
    return sr


def find_retread_hc_events(staff):
    """Identify HCs at team B in season S who had prior HC tenure at team A in
    earlier seasons within window. Return list of (HC_name, prior_team, prior_season,
    new_team, new_season) tuples."""
    s = staff.copy().sort_values(["HC","season"]).reset_index(drop=True)
    events = []
    # For each HC, look at all team-season records they hold
    for hc, g in s.groupby("HC"):
        g = g.sort_values("season").reset_index(drop=True)
        if len(g) < 2:
            continue
        # Find points where team changes
        for i in range(1, len(g)):
            if g.loc[i, "team_nfl"] != g.loc[i-1, "team_nfl"]:
                events.append({
                    "HC": hc,
                    "prior_team": g.loc[i-1, "team_nfl"],
                    "prior_season": int(g.loc[i-1, "season"]),
                    "new_team": g.loc[i, "team_nfl"],
                    "new_season": int(g.loc[i, "season"]),
                })
    return pd.DataFrame(events)


def compute_profile(sr, team, season, k, anchor="last"):
    """Return mean pass_rate_residual for team in season, anchored at last
    or first k games."""
    g = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week")
    if len(g) < k:
        return np.nan
    sub = g.iloc[-k:] if anchor == "last" else g.iloc[:k]
    return float(sub["pass_rate_residual"].mean())


def run_predictive_test(events, sr):
    """For each retread event, compute observed shift, predicted shift, and
    other context. Returns DataFrame of paired observations."""
    rows = []
    for _, ev in events.iterrows():
        hc = ev["HC"]
        prior_team, prior_season = ev["prior_team"], ev["prior_season"]
        new_team, new_season = ev["new_team"], ev["new_season"]
        if pd.isna(prior_team) or pd.isna(new_team):
            continue
        # 1. HC_X's profile at prior team (last K games of prior season)
        profile_hc_prior = compute_profile(sr, prior_team, prior_season, K, anchor="last")
        # 2. New team's profile under prior HC (last K games of season before new HC arrives)
        profile_new_pre = compute_profile(sr, new_team, new_season - 1, K, anchor="last")
        # 3. New team's profile under HC_X (first K games of new season)
        profile_new_post = compute_profile(sr, new_team, new_season, K, anchor="first")

        if any(np.isnan([profile_hc_prior, profile_new_pre, profile_new_post])):
            continue

        shift_obs = profile_new_post - profile_new_pre
        shift_pred = profile_hc_prior - profile_new_pre

        rows.append({
            "HC": hc,
            "prior_team": prior_team, "prior_season": prior_season,
            "new_team": new_team, "new_season": new_season,
            "profile_hc_prior": profile_hc_prior,
            "profile_new_pre": profile_new_pre,
            "profile_new_post": profile_new_post,
            "shift_obs": shift_obs,
            "shift_pred": shift_pred,
        })
    return pd.DataFrame(rows)


def permutation_test_corr(x, y, n_perm=N_PERM_TEST, seed=42):
    """Test whether corr(x, y) is greater than null by shuffling y."""
    rng = np.random.default_rng(seed)
    obs_r = float(np.corrcoef(x, y)[0, 1])
    null_rs = np.zeros(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        null_rs[i] = float(np.corrcoef(x, y_shuf)[0, 1])
    p_one = (null_rs >= obs_r).mean()
    p_two = (np.abs(null_rs) >= abs(obs_r)).mean()
    return obs_r, p_one, p_two, null_rs


def fit_ols(x, y):
    """OLS shift_obs ~ shift_pred. Returns (intercept, slope, R^2)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(beta[0]), float(beta[1]), float(r2)


def main():
    print(f"=== HC PREDICTIVE VALIDATION (K={K} window, n_perm={N_PERM_TEST}) ===\n")
    staff = build_staff_panel()
    print(f"staff rows: {len(staff):,}")

    events = find_retread_hc_events(staff)
    print(f"retread HC events (HC moved team within window): {len(events)}")

    sr = load_offensive_residuals()
    print(f"per-team-game offensive residuals: {len(sr):,}")

    print("\nComputing observed + predicted shifts per retread event...")
    paired = run_predictive_test(events, sr)
    print(f"events with all 3 profiles computable: {len(paired)}")

    if len(paired) < 5:
        print("INSUFFICIENT EVENTS for predictive validation"); return

    # Core test
    x = paired["shift_pred"].values
    y = paired["shift_obs"].values
    obs_r, p_one, p_two, null_rs = permutation_test_corr(x, y)
    intercept, slope, r2 = fit_ols(x, y)

    print(f"\n=== HEADLINE RESULT ===")
    print(f"  n events: {len(paired)}")
    print(f"  Pearson r(shift_pred, shift_obs):  {obs_r:+.4f}")
    print(f"  p_one_sided (perm null):           {p_one:.4f}")
    print(f"  p_two_sided (perm null):           {p_two:.4f}")
    print(f"  OLS slope (calibration):           {slope:+.4f}  (1.0 = perfect; 0 = no signal)")
    print(f"  OLS intercept:                     {intercept:+.4f}")
    print(f"  OLS R^2:                           {r2:.4f}")
    print(f"  null r mean:                       {null_rs.mean():+.4f}")
    print(f"  null r 95th percentile:            {np.percentile(null_rs, 95):+.4f}")

    # Direction agreement
    paired["pred_pos"] = paired["shift_pred"] > 0
    paired["obs_pos"] = paired["shift_obs"] > 0
    agree = (paired["pred_pos"] == paired["obs_pos"]).sum()
    print(f"\n  direction agreement: {agree}/{len(paired)} ({100*agree/len(paired):.0f}%)")
    print(f"  expected by chance:   {len(paired)/2:.0f}")

    # Verdict
    if p_one < 0.05 and obs_r > 0.2:
        verdict = "PORTABLE — HC profile forecasts shift; forecastability gate PASSES"
    elif p_one < 0.05:
        verdict = "WEAK PORTABLE — significant but small effect"
    elif obs_r > 0.2:
        verdict = "MARGINAL — directional but unconfirmed"
    else:
        verdict = "NOT PORTABLE — HC + roster co-form; latent is team-adapted"
    print(f"\n  VERDICT: {verdict}")

    # Save artifacts
    paired.to_csv(ROOT / "hc_predictive_validation_paired.csv", index=False)
    summary = pd.DataFrame([{
        "test": "HC pass_rate portability",
        "n_events": len(paired),
        "pearson_r": obs_r,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "ols_slope": slope,
        "ols_r2": r2,
        "direction_agreement": agree / len(paired),
        "verdict": verdict,
    }])
    summary.to_csv(ROOT / "hc_predictive_validation_summary.csv", index=False)
    print(f"\n  wrote paired + summary CSVs")

    # Top 5 best-predicted + worst-predicted
    paired["abs_resid"] = (paired["shift_obs"] - (intercept + slope * paired["shift_pred"])).abs()
    print(f"\n  Top 5 retread events by absolute observed shift:")
    print(paired.nlargest(5, "shift_obs")[
        ["HC","prior_team","prior_season","new_team","new_season",
         "shift_pred","shift_obs"]
    ].to_string(index=False))
    print(f"\n  Bottom 5 (largest negative shifts):")
    print(paired.nsmallest(5, "shift_obs")[
        ["HC","prior_team","prior_season","new_team","new_season",
         "shift_pred","shift_obs"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
