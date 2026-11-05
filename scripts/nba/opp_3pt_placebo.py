"""NBA placebo baseline for the opponent-three-point-rate portability slope.

Mirrors Block (i) of scripts/predictive_validation_robustness.py (the NFL HC/OC placebo) on the NBA inputs of
scripts/nba/opp_3pt_predictive_validation.py (METRIC = opp_three_pt_rate_residual, K = 10).

Continuous-staff placebo:
  Same head coach for the same team in season S-1 and season S. shift_obs computed within team using
  last-10-of-S-1 -> first-10-of-S. Placebo shift_pred substitutes a random retread coach's prior
  (profile_hc_prior of the observed retread events). Two pools, both reported:
    (a) same-season pool (keyed by the retread event's new_season) with an all-seasons fallback (NFL-identical)
    (b) all-seasons pool only
Residuals, staff and profiles are imported from the validation script, not copied. Reads the paired CSV that
opp_3pt_predictive_validation.py writes in the same run, and writes opp_3pt_placebo.json next to it.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opp_3pt_predictive_validation import (K, METRIC, RESULTS, build_residuals, compute_profile,  # noqa: E402
                                           find_retread, fit_ols, load_staff)

REPO = Path(__file__).resolve().parents[2]
OUT = RESULTS
MODE = os.environ.get("TWOLOCUS_MODE", "submitted")                  # set by reproduce.sh
B = 1000
SEED = 42


def prev_season(s):
    y = int(s.split("-")[0])
    return f"{y - 1}-{str(y % 100).zfill(2)}"


def observed_events(staff, resid):
    """The validation script's 24 events, with profile_hc_prior kept (the paired CSV does not store it)."""
    rows = []
    for _, ev in find_retread(staff).iterrows():
        p_prior = compute_profile(resid, ev["prior_team"], ev["prior_season"], K, "last")
        p_pre = compute_profile(resid, ev["new_team"], prev_season(ev["new_season"]), K, "last")
        p_post = compute_profile(resid, ev["new_team"], ev["new_season"], K, "first")
        if any(np.isnan([p_prior, p_pre, p_post])):
            continue
        rows.append({"HC": ev["HC"], "prior_team": ev["prior_team"], "prior_season": ev["prior_season"],
                     "new_team": ev["new_team"], "new_season": ev["new_season"],
                     "shift_obs": p_post - p_pre, "shift_pred": p_prior - p_pre, "profile_hc_prior": p_prior})
    return pd.DataFrame(rows)


def continuous_staff_seasons(staff):
    """(team, season) where the same HC held the job in season-1 and season (consecutive seasons)."""
    s = staff.sort_values(["team_abbr", "season"]).reset_index(drop=True)
    rows = []
    for tm, g in s.groupby("team_abbr"):
        g = g.sort_values("season").reset_index(drop=True)
        for i in range(1, len(g)):
            prev, cur = g.loc[i - 1], g.loc[i]
            if pd.isna(prev["HC"]) or pd.isna(cur["HC"]):
                continue
            if prev["HC"] == cur["HC"] and prev["season"] == prev_season(cur["season"]):
                rows.append({"team_abbr": tm, "person": cur["HC"], "prior_season": prev["season"],
                             "new_season": cur["season"]})
    return pd.DataFrame(rows)


def placebo(cont_df, pools, broad, rng):
    slopes = np.empty(B)
    pre, obs = cont_df["profile_pre"].values, cont_df["shift_obs_placebo"].values
    for b in range(B):
        sampled = []
        for _, row in cont_df.iterrows():
            pool = pools.get(row["new_season"], broad) if pools is not None else broad
            if len(pool) == 0:
                pool = broad
            sampled.append(rng.choice(pool))
        _, slopes[b], _ = fit_ols(np.array(sampled) - pre, obs)
    return slopes


def summarise(slopes, observed):
    med = float(np.median(slopes))
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return {"placebo_median": med, "placebo_95ci": [float(lo), float(hi)],
            "observed_percentile": float((slopes < observed).mean()), "excess": float(observed - med)}


def main():
    paired_csv = pd.read_csv(OUT / "opp_3pt_predictive_validation_paired.csv")
    _, obs_slope, _ = fit_ols(paired_csv["shift_pred"].values, paired_csv["shift_obs"].values)
    print(f"observed slope (paired CSV): {obs_slope:+.4f}  n: {len(paired_csv)}")
    if MODE == "submitted" and round(obs_slope, 2) != 0.56:              # the paper's slope; corrected differs
        raise SystemExit(f"STOP: observed slope {obs_slope:.4f} does not match 0.56")

    staff, resid = load_staff(), build_residuals()
    ev = observed_events(staff, resid)
    key = ["HC", "prior_team", "prior_season", "new_team", "new_season"]
    m = ev.merge(paired_csv, on=key, suffixes=("", "_csv"))
    assert len(ev) == len(paired_csv) == len(m), (len(ev), len(paired_csv), len(m))
    assert np.allclose(m.shift_pred, m.shift_pred_csv) and np.allclose(m.shift_obs, m.shift_obs_csv)
    print(f"rebuilt events match the paired CSV: {len(m)}/{len(paired_csv)}")

    cont = continuous_staff_seasons(staff)
    rows, dropped = [], []
    for _, r in cont.iterrows():
        pre = compute_profile(resid, r["team_abbr"], r["prior_season"], K, "last")
        post = compute_profile(resid, r["team_abbr"], r["new_season"], K, "first")
        if np.isnan(pre) or np.isnan(post):
            dropped.append(f"{r['team_abbr']} {r['new_season']}")
            continue
        rows.append({"team_abbr": r["team_abbr"], "new_season": r["new_season"], "profile_pre": pre,
                     "profile_post": post, "shift_obs_placebo": post - pre})
    cont_df = pd.DataFrame(rows)
    print(f"continuous-staff candidates {len(cont)}, computable {len(cont_df)}, dropped {len(dropped)}")

    pools = {s: ev[ev["new_season"] == s]["profile_hc_prior"].values for s in sorted(ev["new_season"].unique())}
    broad = ev["profile_hc_prior"].values
    sizes = [len(v) for v in pools.values()]
    n_fallback = int((~cont_df["new_season"].isin(pools.keys())).sum())
    res = {}
    for name, p in (("a_same_season_with_fallback", pools), ("b_all_seasons", None)):
        res[name] = summarise(placebo(cont_df, p, broad, np.random.default_rng(SEED)), obs_slope)
        print(name, res[name])

    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    out = {"mode": MODE, "observed_slope": obs_slope, "n_events": int(len(paired_csv)),
           "n_continuous_staff_candidates": int(len(cont)), "n_continuous_staff_team_seasons": int(len(cont_df)),
           "continuous_staff_dropped": dropped,
           "pool_sizes_same_season": {"min": min(sizes), "median": float(np.median(sizes)), "max": max(sizes),
                                      "n_seasons": len(sizes)},
           "n_rows_using_fallback_pool_in_a": n_fallback,
           **res, "B": B, "K": K, "seed": SEED, "metric": METRIC,
           "rng": "numpy default_rng(seed), a fresh generator per pool variant",
           "script": "scripts/nba/opp_3pt_placebo.py", "git_commit": commit}
    (OUT / "opp_3pt_placebo.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT / "opp_3pt_placebo.json")


if __name__ == "__main__":
    main()
