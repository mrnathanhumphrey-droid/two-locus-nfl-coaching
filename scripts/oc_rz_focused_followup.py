"""RZ-focused OC follow-up — power up the marginal RZ trips signal.

OC-change study showed RZ trips 1.11x p=0.075 (just under 0.05 gate). Mechanism
is plausible (RZ game-planning is detailed OC work).

This test powers up:
  - Wider window: K=8 games (vs K=6 in main test)
  - Additional RZ-specific metrics: RZ pass-rate, RZ TD-rate, RZ shotgun-rate
  - Same OC-change-with-HC-continuity design as main test

If RZ pass-rate or RZ TD-rate carries at OC level, it'd be the only positive
OC signal — and would point to "OC owns RZ game-planning" as a specific claim.
If null on all four targets → RZ trips marginal was noise, no OC signal lives
in this layer.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_change_event_study import (
    PFR_TO_NFL, LEGACY_REMAP, build_staff_panel, classify_offseason_events,
    permutation_test, N_PERM,
)

ROOT = Path(__file__).resolve().parent.parent / "data"
K = 8  # wider window for power


def build_rz_residuals():
    """Per-game red-zone-specific metrics from pbp.

    For each (game, posteam) compute:
      - rz_pass_rate: fraction of RZ plays that were pass
      - rz_td_rate: fraction of RZ trips ending in TD
      - rz_shotgun_rate: fraction of RZ plays in shotgun
    """
    cols = ["game_id","play_id","season","week","posteam","play_type",
            "yardline_100","shotgun","qb_kneel","qb_spike","two_point_attempt",
            "touchdown","drive"]
    pbp = pd.read_parquet(ROOT / "games" / "pbp.parquet", columns=cols)
    pbp = pbp[(pbp["play_type"].isin(["pass","run"]))
              & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)
              & (pbp["two_point_attempt"] != 1)
              & pbp["posteam"].notna()].copy()
    rz = pbp[pbp["yardline_100"] <= 20].copy()
    rz["is_pass"] = (rz["play_type"] == "pass").astype(int)
    rz["is_shotgun"] = rz["shotgun"].fillna(0).astype(int)
    rz["is_td"] = rz["touchdown"].fillna(0).astype(int)

    # Per-team-game RZ pass rate + shotgun rate
    tg = (rz.groupby(["game_id","season","week","posteam"]).agg(
        rz_plays=("play_id","count"),
        rz_pass_rate=("is_pass","mean"),
        rz_shotgun_rate=("is_shotgun","mean"),
    ).reset_index())
    # Per-team-game RZ-trip TD rate (group by drive, was there a TD in any RZ play of that drive)
    drive_td = (rz.groupby(["game_id","season","week","posteam","drive"])["is_td"].max()
                  .reset_index().rename(columns={"is_td":"drive_had_td"}))
    rz_td = (drive_td.groupby(["game_id","season","week","posteam"]).agg(
        rz_trips=("drive","nunique"),
        rz_td_count=("drive_had_td","sum"),
    ).reset_index())
    rz_td["rz_td_rate"] = rz_td["rz_td_count"] / rz_td["rz_trips"]

    rz_panel = tg.merge(rz_td[["game_id","posteam","rz_trips","rz_td_rate"]],
                          on=["game_id","posteam"], how="left")
    rz_panel = rz_panel[rz_panel["rz_plays"] >= 3]  # need at least 3 RZ plays to compute

    # Compute residuals: subtract season-level league mean per metric
    for met in ["rz_pass_rate","rz_shotgun_rate","rz_td_rate"]:
        season_mean = rz_panel.groupby("season")[met].transform("mean")
        rz_panel[f"{met}_residual"] = rz_panel[met] - season_mean

    rz_panel["team_canon"] = rz_panel["posteam"].map(LEGACY_REMAP).fillna(rz_panel["posteam"])
    return rz_panel


def compute_pre_post(sr, team, season, metric, k=K):
    prev = sr[(sr["team_canon"] == team) & (sr["season"] == season-1)].sort_values("week")
    curr = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week")
    if len(prev) < k or len(curr) < k:
        return np.nan
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    return float(post - pre)


def main():
    print(f"=== RZ-FOCUSED OC FOLLOW-UP (K={K} games per window) ===\n")
    staff = build_staff_panel()
    classified = classify_offseason_events(staff)
    print(f"OC event class dist: {classified['event_class'].value_counts().to_dict()}")

    print("\nBuilding RZ-specific residuals...")
    rz_panel = build_rz_residuals()
    print(f"  RZ team-game rows: {len(rz_panel):,}")
    print(f"  seasons: {rz_panel['season'].min()}-{rz_panel['season'].max()}")

    metrics = ["rz_pass_rate_residual","rz_shotgun_rate_residual","rz_td_rate_residual"]
    rows = []
    for _, ev in classified.iterrows():
        if ev["event_class"] not in ("OC_CHANGE_HC_SAME","CONTINUOUS"):
            continue
        tm, ssn = ev["team_nfl"], int(ev["season"])
        for met in metrics:
            d = compute_pre_post(rz_panel, tm, ssn, met)
            if np.isnan(d):
                continue
            rows.append({
                "team": tm, "season": ssn, "metric": met,
                "delta": d, "abs_delta": abs(d),
                "is_real_change": ev["event_class"] == "OC_CHANGE_HC_SAME",
            })
    panel = pd.DataFrame(rows)
    n_real = panel[panel["is_real_change"]].groupby("metric").size().to_dict()
    n_plac = panel[~panel["is_real_change"]].groupby("metric").size().to_dict()
    print(f"\n  real OC-change events per metric: {n_real}")
    print(f"  placebo (continuous staff) events per metric: {n_plac}\n")

    print(f"  {'metric':<30} {'|d|_real':>10} {'|d|_plac':>10} {'ratio':>7} "
          f"{'p_perm':>8} {'verdict':>25}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*25}")
    summary = []
    for met in metrics:
        real = panel[(panel["metric"] == met) & panel["is_real_change"]]["abs_delta"].values
        plac = panel[(panel["metric"] == met) & ~panel["is_real_change"]]["abs_delta"].values
        if len(real) == 0 or len(plac) == 0:
            print(f"  {met:<30} insufficient data"); continue
        r_mean, p_mean = real.mean(), plac.mean()
        ratio = r_mean / p_mean if p_mean > 0 else np.nan
        obs_diff, p_val = permutation_test(real, plac)
        verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
                   else "NO DETECTABLE SHIFT" if p_val >= 0.1 else "MARGINAL")
        summary.append({"metric": met, "n_real": len(real), "n_plac": len(plac),
                        "abs_d_real_mean": r_mean, "abs_d_plac_mean": p_mean,
                        "ratio": ratio, "p_perm": p_val, "verdict": verdict})
        print(f"  {met:<30} {r_mean:>10.4f} {p_mean:>10.4f} {ratio:>7.3f} "
              f"{p_val:>8.4f} {verdict:>25}")

    pd.DataFrame(summary).to_csv(ROOT / "oc_rz_focused_summary.csv", index=False)
    panel.to_csv(ROOT / "oc_rz_focused_panel.csv", index=False)
    print(f"\n  wrote summary + panel CSVs")

    print(f"\n=== Reference: OC-change study main result ===")
    print(f"  rz_trips_residual: 1.11x, p=0.075 (MARGINAL, K=6 window, generic metric)")


if __name__ == "__main__":
    main()
