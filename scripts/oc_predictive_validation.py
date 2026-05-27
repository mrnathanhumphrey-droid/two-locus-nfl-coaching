"""OC predictive validation — does incoming OC's prior RZ shotgun-rate
profile forecast the new team's RZ-shotgun-rate shift at the OC-change event?

Parallel to HC predictive validation but targets the OC-specific finding:
  Layer 2d found OC change → RZ shotgun-rate shift 1.34x p=0.0002.
This test asks whether that latent is FORECASTABLE from the OC's prior data.

Design (retread OCs only):
  For each event where OC_X moves from team A to team B at offseason S:
    profile_OC_X_at_A: rz_shotgun_rate mean over OC_X's last K games at A
    profile_B_under_prior: rz_shotgun_rate mean over team B's last K games
                            of season S-1 (under prior OC at B)
    profile_B_under_X: rz_shotgun_rate mean over team B's first K games
                       of season S (under OC_X)

  Observed shift:   shift_obs = profile_B_under_X - profile_B_under_prior
  Predicted shift:  shift_pred = profile_OC_X_at_A - profile_B_under_prior

  Same correlation + OLS + permutation test as HC version.

Key difference from HC: we partial out HC continuity. If HC also changed
between seasons (rare for OC moves to happen with HC stable, but possible),
that's not a clean OC isolation. We include all retread OCs but flag.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_change_event_study import (
    PFR_TO_NFL, LEGACY_REMAP, build_staff_panel,
)
from oc_rz_focused_followup import build_rz_residuals
from hc_predictive_validation import (
    compute_profile, permutation_test_corr, fit_ols, N_PERM_TEST, K,
)

ROOT = Path(__file__).resolve().parent.parent / "data"


def find_retread_oc_events(staff):
    """Find OCs who moved between teams within our window."""
    s = staff.copy().sort_values(["OC","season"]).reset_index(drop=True)
    events = []
    for oc, g in s.groupby("OC"):
        if oc is None or pd.isna(oc) or oc == "":
            continue
        g = g.sort_values("season").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(1, len(g)):
            if g.loc[i, "team_nfl"] != g.loc[i-1, "team_nfl"]:
                events.append({
                    "OC": oc,
                    "prior_team": g.loc[i-1, "team_nfl"],
                    "prior_season": int(g.loc[i-1, "season"]),
                    "new_team": g.loc[i, "team_nfl"],
                    "new_season": int(g.loc[i, "season"]),
                })
    return pd.DataFrame(events)


def compute_oc_profile(rz, team, season, k, anchor="last"):
    """Mean rz_shotgun_rate_residual for team-season, last/first k games."""
    g = rz[(rz["team_canon"] == team) & (rz["season"] == season)].sort_values("week")
    if len(g) < k:
        return np.nan
    sub = g.iloc[-k:] if anchor == "last" else g.iloc[:k]
    return float(sub["rz_shotgun_rate_residual"].mean())


def run_predictive_test_oc(events, rz):
    rows = []
    for _, ev in events.iterrows():
        oc = ev["OC"]
        prior_team, prior_season = ev["prior_team"], ev["prior_season"]
        new_team, new_season = ev["new_team"], ev["new_season"]
        if pd.isna(prior_team) or pd.isna(new_team):
            continue
        p_prior = compute_oc_profile(rz, prior_team, prior_season, K, anchor="last")
        p_pre = compute_oc_profile(rz, new_team, new_season - 1, K, anchor="last")
        p_post = compute_oc_profile(rz, new_team, new_season, K, anchor="first")
        if any(np.isnan([p_prior, p_pre, p_post])):
            continue
        shift_obs = p_post - p_pre
        shift_pred = p_prior - p_pre
        rows.append({
            "OC": oc,
            "prior_team": prior_team, "prior_season": prior_season,
            "new_team": new_team, "new_season": new_season,
            "profile_oc_prior": p_prior,
            "profile_new_pre": p_pre,
            "profile_new_post": p_post,
            "shift_obs": shift_obs,
            "shift_pred": shift_pred,
        })
    return pd.DataFrame(rows)


def main():
    print(f"=== OC PREDICTIVE VALIDATION on RZ shotgun-rate (K={K}, n_perm={N_PERM_TEST}) ===\n")
    staff = build_staff_panel()
    events = find_retread_oc_events(staff)
    print(f"retread OC events (OC moved team within window): {len(events)}")

    print("\nBuilding RZ residuals (offensive schematic)...")
    rz = build_rz_residuals()
    print(f"  RZ team-game rows: {len(rz):,}")

    print("\nComputing observed + predicted shifts per retread event...")
    paired = run_predictive_test_oc(events, rz)
    print(f"events with all 3 profiles computable: {len(paired)}")

    if len(paired) < 5:
        print("INSUFFICIENT EVENTS for predictive validation"); return

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
    print(f"  null r 95th pct:                   {np.percentile(null_rs, 95):+.4f}")

    paired["pred_pos"] = paired["shift_pred"] > 0
    paired["obs_pos"] = paired["shift_obs"] > 0
    agree = (paired["pred_pos"] == paired["obs_pos"]).sum()
    print(f"\n  direction agreement: {agree}/{len(paired)} ({100*agree/len(paired):.0f}%)")
    print(f"  expected by chance:   {len(paired)/2:.0f}")

    if p_one < 0.05 and obs_r > 0.2:
        verdict = "PORTABLE — OC profile forecasts RZ shotgun shift; forecastability gate PASSES"
    elif p_one < 0.05:
        verdict = "WEAK PORTABLE — significant but small effect"
    elif obs_r > 0.2:
        verdict = "MARGINAL — directional but unconfirmed"
    else:
        verdict = "NOT PORTABLE — OC + roster co-form RZ identity; latent is team-adapted"
    print(f"\n  VERDICT: {verdict}")

    paired.to_csv(ROOT / "oc_predictive_validation_paired.csv", index=False)
    pd.DataFrame([{
        "test": "OC rz_shotgun_rate portability",
        "n_events": len(paired),
        "pearson_r": obs_r,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "ols_slope": slope,
        "ols_r2": r2,
        "direction_agreement": agree / len(paired),
        "verdict": verdict,
    }]).to_csv(ROOT / "oc_predictive_validation_summary.csv", index=False)
    print(f"\n  wrote paired + summary CSVs")

    print(f"\n  Top 5 retread OC events by absolute observed RZ shotgun shift:")
    print(paired.nlargest(5, "shift_obs")[
        ["OC","prior_team","prior_season","new_team","new_season","shift_pred","shift_obs"]
    ].to_string(index=False))
    print(f"\n  Bottom 5 (largest negative shifts):")
    print(paired.nsmallest(5, "shift_obs")[
        ["OC","prior_team","prior_season","new_team","new_season","shift_pred","shift_obs"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
