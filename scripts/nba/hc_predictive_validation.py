"""NBA HC predictive validation — does incoming HC's prior pace-residual
profile forecast the new team's pace shift?

Mirrors NFL HC predictive validation methodology. Tests forecastability of
the cross-league Layer 2 finding (HC change -> pace shift).

Design:
  Retread events: HCs who moved between teams within window.
  For each event where HC_X moves from team A in season S-1 to team B in S:
    profile_HC_X_at_A: pace_residual mean over HC_X's last K games at A
    profile_B_under_prior: pace_residual mean over B's last K games of S-1
    profile_B_under_X: pace_residual mean over B's first K games of S
  Observed shift: post - pre
  Predicted shift: profile_HC_X_at_A - profile_B_under_prior
  Test: Pearson r(predicted, observed), OLS slope, permutation null B=5000

Verdict gate: r > 0.20 AND p_one_sided < 0.05 -> PORTABLE.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "nba" / "inputs"   # staged per mode: scripts/build/stage_nba.py
RESID = ROOT / "data" / "parquet" / "team_game_residuals.parquet"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

K = 10
N_PERM = 5000
SEED = 42
METRIC = "possessions_residual"


def load_coach_team_season() -> pd.DataFrame:
    """Of-record HC per team-season (start-of-season coach)."""
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    return flags[["team_abbr", "season", "season_end_year",
                   "start_of_season_name", "start_of_season_slug",
                   "of_record_coach_name"]].rename(
        columns={"start_of_season_name": "HC"})


def find_retread_hc_events(staff: pd.DataFrame) -> pd.DataFrame:
    """HCs who appear at >=2 different teams in window."""
    s = staff[staff["HC"].notna()].copy().sort_values(["HC", "season"]).reset_index(drop=True)
    events = []
    for hc, g in s.groupby("HC"):
        g = g.sort_values("season").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(1, len(g)):
            if g.loc[i, "team_abbr"] != g.loc[i-1, "team_abbr"]:
                events.append({
                    "HC": hc,
                    "prior_team": g.loc[i-1, "team_abbr"],
                    "prior_season": g.loc[i-1, "season"],
                    "new_team": g.loc[i, "team_abbr"],
                    "new_season": g.loc[i, "season"],
                })
    return pd.DataFrame(events)


def compute_profile(resid: pd.DataFrame, team: str, season_str: str,
                    k: int, anchor: str = "last") -> float:
    g = resid[(resid["team_abbr"] == team)
              & (resid["season"] == season_str)].sort_values("game_in_season")
    if len(g) < k:
        return np.nan
    sub = g.iloc[-k:] if anchor == "last" else g.iloc[:k]
    return float(sub[METRIC].mean())


def permutation_test_corr(x, y, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs_r = float(np.corrcoef(x, y)[0, 1])
    null_rs = np.zeros(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        null_rs[i] = float(np.corrcoef(x, y_shuf)[0, 1])
    p_one = float((null_rs >= obs_r).mean())
    p_two = float((np.abs(null_rs) >= abs(obs_r)).mean())
    return obs_r, p_one, p_two, null_rs


def fit_ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(beta[0]), float(beta[1]), float(r2)


def main():
    print(f"=== NBA HC PREDICTIVE VALIDATION (metric={METRIC}, K={K}) ===\n")
    staff = load_coach_team_season()
    print(f"team-seasons in staff panel: {len(staff):,}")
    events = find_retread_hc_events(staff)
    print(f"retread HC events (HC moved team within window): {len(events)}")
    print(events.to_string(index=False))

    resid = pd.read_parquet(RESID)

    rows = []
    for _, ev in events.iterrows():
        p_prior = compute_profile(resid, ev["prior_team"], ev["prior_season"], K, "last")
        p_pre = compute_profile(resid, ev["new_team"], ev["prior_season"], K, "last")
        # Wait: pre window should be new_team in season BEFORE new_season
        # new_season "2023-24" -> prior is "2022-23"; but the new team's prior staff
        # operated in new_season - 1 = "2022-23"
        # Equal to prior_season only if no gap; otherwise need new_season - 1
        new_y = int(ev["new_season"].split("-")[0])
        new_pre_season_str = f"{new_y - 1}-{str(new_y % 100).zfill(2)}"
        p_pre = compute_profile(resid, ev["new_team"], new_pre_season_str, K, "last")
        p_post = compute_profile(resid, ev["new_team"], ev["new_season"], K, "first")
        if any(np.isnan([p_prior, p_pre, p_post])):
            continue
        shift_obs = p_post - p_pre
        shift_pred = p_prior - p_pre
        rows.append({
            "HC": ev["HC"],
            "prior_team": ev["prior_team"], "prior_season": ev["prior_season"],
            "new_team": ev["new_team"], "new_season": ev["new_season"],
            "profile_hc_prior": p_prior, "profile_new_pre": p_pre,
            "profile_new_post": p_post,
            "shift_obs": shift_obs, "shift_pred": shift_pred,
        })
    paired = pd.DataFrame(rows)
    print(f"\nevents with all 3 profiles computable: {len(paired)}")

    if len(paired) < 5:
        print("INSUFFICIENT EVENTS for predictive validation"); return

    x = paired["shift_pred"].values
    y = paired["shift_obs"].values
    obs_r, p_one, p_two, null_rs = permutation_test_corr(x, y)
    intercept, slope, r2 = fit_ols(x, y)
    paired["pred_pos"] = paired["shift_pred"] > 0
    paired["obs_pos"] = paired["shift_obs"] > 0
    agree = int((paired["pred_pos"] == paired["obs_pos"]).sum())

    print(f"\n=== HEADLINE ===")
    print(f"  n events:                          {len(paired)}")
    print(f"  Pearson r(shift_pred, shift_obs):  {obs_r:+.4f}")
    print(f"  p_one_sided (perm null):           {p_one:.4f}")
    print(f"  p_two_sided (perm null):           {p_two:.4f}")
    print(f"  OLS slope (calibration):           {slope:+.4f}  (1.0 = perfect; 0 = no signal)")
    print(f"  OLS intercept:                     {intercept:+.4f}")
    print(f"  OLS R^2:                           {r2:.4f}")
    print(f"  direction agreement:               {agree}/{len(paired)} ({100*agree/len(paired):.0f}%)")

    if p_one < 0.05 and obs_r > 0.2:
        verdict = "PORTABLE - HC pace profile forecasts shift"
    elif p_one < 0.05:
        verdict = "WEAK PORTABLE"
    elif obs_r > 0.2:
        verdict = "MARGINAL - directional but unconfirmed"
    else:
        verdict = "NOT PORTABLE - pace identity is team-adapted"
    print(f"  VERDICT: {verdict}")

    print(f"\n  Cross-reference NFL HC predictive validation:")
    print(f"  n=41 retread events, r=+0.386, p=0.004, slope=0.30, dir agree 59%")

    paired.to_csv(RESULTS / "hc_predictive_validation_paired.csv", index=False)
    summary = pd.DataFrame([{
        "test": "NBA HC pace portability",
        "n_events": len(paired),
        "pearson_r": obs_r, "p_one_sided": p_one, "p_two_sided": p_two,
        "ols_slope": slope, "ols_r2": r2,
        "direction_agreement": agree / len(paired),
        "verdict": verdict,
    }])
    summary.to_csv(RESULTS / "hc_predictive_validation_summary.csv", index=False)
    print(f"\n  wrote paired + summary CSVs to {RESULTS}")

    if len(paired) >= 5:
        print(f"\n  Top 5 retread events by absolute observed shift:")
        print(paired.nlargest(5, paired["shift_obs"].abs().name
                                if hasattr(paired["shift_obs"].abs(), "name")
                                else "shift_obs")[["HC","prior_team","prior_season",
                                                     "new_team","new_season",
                                                     "shift_pred","shift_obs"]]
                .to_string(index=False))


if __name__ == "__main__":
    main()
