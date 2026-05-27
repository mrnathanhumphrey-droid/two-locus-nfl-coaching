"""OC-change event study — parallel to HC-change Layer 2 test.

Design:
  Treatment events: offseason OC changes with HC CONTINUITY (isolates OC
  effect; HC same in S-1 and S, OC different).

  Placebo events: continuous staff (same HC AND same OC in S-1 and S).

  Window: K=6 games. Pre = last K games of season S-1. Post = first K games
  of season S. Target: |mean(post) - mean(pre)| per metric.

  Comparison: real |Δ| vs placebo |Δ| via 5000-perm null on mean difference.

  Why offseason-only: mid-season OC changes are very rare and not detectable
  from coach_seasons (games-played field is all NaN for OCs in this panel).
  Offseason changes are also the cleaner natural experiment — planned,
  not crisis-driven.

  Confounds we accept:
  - Roster/draft turnover between seasons (we don't try to isolate)
  - League-wide schematic drift across years (residuals are within-season-baselined)
  - QB changes that often accompany OC changes (not separated in this pass)

Same SNR discipline as HC test: real vs placebo via 5000-perm null on
mean(|Δ|) difference. p < 0.05 + ratio > 1.1 -> "STAFF SHIFT MOVES NEEDLE".
"""
from __future__ import annotations
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"
K = 6  # games per window
N_PERM = 5000
SEED = 42


PFR_TO_NFL = {
    "rai": "LV", "rav": "BAL", "oti": "TEN", "sdg": "LAC", "htx": "HOU",
    "clt": "IND", "crd": "ARI", "ram": "LA", "buf": "BUF", "mia": "MIA",
    "nwe": "NE", "nyj": "NYJ", "cin": "CIN", "cle": "CLE", "pit": "PIT",
    "jax": "JAX", "den": "DEN", "kan": "KC", "dal": "DAL", "nyg": "NYG",
    "phi": "PHI", "was": "WAS", "chi": "CHI", "det": "DET", "gnb": "GB",
    "min": "MIN", "atl": "ATL", "car": "CAR", "nor": "NO", "tam": "TB",
    "sea": "SEA", "sfo": "SF",
}
LEGACY_REMAP = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA"}


def build_staff_panel():
    """Pivot coach_seasons to (team_nfl, season) -> HC, OC names.

    Uses the primary (most-games or first) coach in each role per team-season.
    Same logic as snr_test_perm_and_event.build_static_df.
    """
    cs = pd.read_parquet(ROOT / "coaching/coach_seasons.parquet")
    cs["games_filled"] = cs["games"].fillna(0)
    primary = (cs[cs["role"].isin(["head_coach","offensive_coordinator"])]
               .sort_values(["team","season","role","games_filled"],
                            ascending=[True, True, True, False])
               .groupby(["team","season","role"]).first().reset_index())
    staff = primary.pivot_table(index=["team","season"], columns="role",
                                  values="coach_name", aggfunc="first").reset_index()
    staff = staff.rename(columns={"head_coach": "HC", "offensive_coordinator": "OC"})
    staff["team_nfl"] = staff["team"].map(PFR_TO_NFL)
    return staff[["team_nfl","season","HC","OC"]].dropna(subset=["team_nfl"])


def classify_offseason_events(staff):
    """For each (team, season) with both S-1 and S records, classify."""
    staff = staff.copy().sort_values(["team_nfl","season"])
    # Prior-year HC/OC on same team
    staff["HC_prev"] = staff.groupby("team_nfl")["HC"].shift(1)
    staff["OC_prev"] = staff.groupby("team_nfl")["OC"].shift(1)
    staff["season_prev"] = staff.groupby("team_nfl")["season"].shift(1)
    # Drop rows without prior or with gap
    s = staff[(staff["season"] - staff["season_prev"]) == 1].copy()
    # Require both HC and OC named
    s = s.dropna(subset=["HC","OC","HC_prev","OC_prev"])
    s["hc_same"] = s["HC"] == s["HC_prev"]
    s["oc_same"] = s["OC"] == s["OC_prev"]
    s["event_class"] = np.where(
        s["hc_same"] & ~s["oc_same"], "OC_CHANGE_HC_SAME",
        np.where(s["hc_same"] & s["oc_same"], "CONTINUOUS",
                 np.where(~s["hc_same"], "HC_CHANGE_EXCLUDE",
                          "OTHER"))
    )
    return s


def load_state_residuals():
    sr = pd.read_parquet(ROOT / "games/state_residuals.parquet")
    if "week" not in sr.columns:
        games = pd.read_parquet(ROOT / "games/games.parquet")
        sr = sr.merge(games[["game_id","week"]], on="game_id", how="left")
    sr["team_canon"] = sr["posteam"].map(LEGACY_REMAP).fillna(sr["posteam"])
    return sr


def compute_pre_post_delta(sr, team, season, metric, k=K):
    """Pre = last k of season-1, post = first k of season. Returns delta or NaN."""
    prev = sr[(sr["team_canon"] == team) & (sr["season"] == season-1)].sort_values("week")
    curr = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week")
    if len(prev) < k or len(curr) < k:
        return np.nan, len(prev), len(curr)
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    return float(post - pre), len(prev), len(curr)


def build_event_panel(staff_classified, sr, metrics):
    """For each treatment + placebo event, compute deltas across metrics."""
    rows = []
    for _, ev in staff_classified.iterrows():
        if ev["event_class"] not in ("OC_CHANGE_HC_SAME", "CONTINUOUS"):
            continue
        tm, ssn = ev["team_nfl"], int(ev["season"])
        for met in metrics:
            d, n_prev, n_curr = compute_pre_post_delta(sr, tm, ssn, met)
            if np.isnan(d):
                continue
            rows.append({
                "team": tm, "season": ssn, "metric": met,
                "delta": d, "abs_delta": abs(d),
                "n_prev": n_prev, "n_curr": n_curr,
                "is_real_change": ev["event_class"] == "OC_CHANGE_HC_SAME",
                "HC": ev["HC"], "OC_prev": ev["OC_prev"], "OC_new": ev["OC"],
                "event_class": ev["event_class"],
            })
    return pd.DataFrame(rows)


def permutation_test(real_abs, plac_abs, n_perm=N_PERM, seed=999):
    """Permutation test on |Δ| means difference."""
    rng = np.random.default_rng(seed)
    obs_diff = real_abs.mean() - plac_abs.mean()
    all_vals = np.concatenate([real_abs, plac_abs])
    n_real = len(real_abs)
    null_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(all_vals)
        null_diffs[i] = shuffled[:n_real].mean() - shuffled[n_real:].mean()
    p = (null_diffs >= obs_diff).mean()
    return obs_diff, p


def main():
    print(f"=== OC-CHANGE EVENT STUDY (K={K} games per window, n_perm={N_PERM}) ===\n")

    print("Building staff panel...")
    staff = build_staff_panel()
    print(f"  staff rows: {len(staff):,}, seasons {staff['season'].min()}-{staff['season'].max()}")

    print("\nClassifying year-over-year events...")
    classified = classify_offseason_events(staff)
    print(f"  with prior-year staff: {len(classified):,}")
    print(f"  event class dist: {classified['event_class'].value_counts().to_dict()}")

    print("\nLoading state_residuals...")
    sr = load_state_residuals()
    print(f"  team-game rows: {len(sr):,}")

    metrics = ["pass_rate_residual", "plays_residual", "red_zone_trips_residual"]
    print(f"\nComputing pre/post deltas for {len(metrics)} metrics...")
    panel = build_event_panel(classified, sr, metrics)
    print(f"  panel rows: {len(panel):,}")
    print(f"  real OC-change events with computable deltas (per metric): "
          f"{panel[panel['is_real_change']].groupby('metric').size().to_dict()}")
    print(f"  placebo (continuous staff) events per metric: "
          f"{panel[~panel['is_real_change']].groupby('metric').size().to_dict()}")

    print(f"\n  {'metric':<26} {'|d|_real':>10} {'|d|_plac':>10} {'ratio':>7} "
          f"{'p_perm':>8} {'verdict':>25}")
    print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*25}")

    summary_rows = []
    for met in metrics:
        real = panel[(panel["metric"] == met) & (panel["is_real_change"])]["abs_delta"].values
        plac = panel[(panel["metric"] == met) & (~panel["is_real_change"])]["abs_delta"].values
        if len(real) == 0 or len(plac) == 0:
            print(f"  {met:<26}  insufficient data")
            continue
        r_mean = real.mean()
        p_mean = plac.mean()
        ratio = r_mean / p_mean if p_mean > 0 else np.nan
        obs_diff, p_val = permutation_test(real, plac)

        verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
                   else "NO DETECTABLE SHIFT" if p_val >= 0.1
                   else "MARGINAL")
        summary_rows.append({
            "metric": met, "n_real": len(real), "n_plac": len(plac),
            "abs_d_real_mean": r_mean, "abs_d_plac_mean": p_mean,
            "ratio": ratio, "p_perm": p_val, "verdict": verdict,
        })
        print(f"  {met:<26} {r_mean:>10.4f} {p_mean:>10.4f} {ratio:>7.3f} "
              f"{p_val:>8.4f} {verdict:>25}")

    summary = pd.DataFrame(summary_rows)
    out_panel = ROOT / "oc_change_event_panel.csv"
    out_summary = ROOT / "oc_change_event_summary.csv"
    panel.to_csv(out_panel, index=False)
    summary.to_csv(out_summary, index=False)
    print(f"\n  wrote {out_panel}")
    print(f"  wrote {out_summary}")

    # Compare directly to HC result for the headline
    print(f"\n=== Cross-reference: HC-change event study (from prior run) ===")
    print(f"  pass_rate_residual:       HC ratio 1.36x, p=0.002 (STAFF SHIFT MOVES NEEDLE)")
    print(f"  plays_residual:           HC ratio 1.10x, p=0.187 (no detectable shift)")
    print(f"  red_zone_trips_residual:  HC ratio 1.07x, p=0.256 (no detectable shift)")


if __name__ == "__main__":
    main()
