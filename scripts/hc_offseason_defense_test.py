"""HC offseason change × DEFENSIVE outcomes — disambiguate personnel-driven
vs HC-owned defense.

The DC RZ-focused test (Layer 2e) returned clean null on opp_rz_pass_rate,
opp_rz_sack_rate, opp_rz_td_rate. Two interpretations:

  1. Defensive RZ identity is PERSONNEL-DRIVEN (DL/LB/DB matchups dominate
     the shrunk field). DCs are reactive, not impositional.

  2. Defensive coordination is HC-OWNED. Macro defensive decisions sit with
     the HC just like macro offensive decisions.

This test distinguishes:
  - If HC change shifts opp_rz_X significantly → interpretation 2 supported
    (HC owns defense at fine-grain just like OC owns offense at fine-grain).
  - If HC change is null on opp_rz_X → interpretation 1 supported
    (defensive identity at RZ is personnel-driven, not coaching-driven).

Design: HC offseason changes (n=168 events) vs continuous staff (n=393)
placebo, K=8 window, same RZ-defensive metrics + generic defensive
metrics for breadth.
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
from dc_change_event_study import build_defensive_residuals
from dc_rz_focused_followup import build_dc_rz_residuals

ROOT = Path(__file__).resolve().parent.parent / "data"
K = 8


def compute_pre_post(sr, team, season, metric, k=K):
    prev = sr[(sr["team_canon"] == team) & (sr["season"] == season-1)].sort_values("week")
    curr = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week")
    if len(prev) < k or len(curr) < k:
        return np.nan
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    if np.isnan(pre) or np.isnan(post):
        return np.nan
    return float(post - pre)


def run_test(panel_residuals, classified, metrics, label):
    """Run HC-offseason-change vs CONTINUOUS placebo on a metric panel."""
    rows = []
    for _, ev in classified.iterrows():
        if ev["event_class"] not in ("HC_CHANGE_EXCLUDE", "CONTINUOUS"):
            continue
        tm, ssn = ev["team_nfl"], int(ev["season"])
        for met in metrics:
            d = compute_pre_post(panel_residuals, tm, ssn, met)
            if np.isnan(d):
                continue
            rows.append({
                "team": tm, "season": ssn, "metric": met,
                "delta": d, "abs_delta": abs(d),
                "is_real_change": ev["event_class"] == "HC_CHANGE_EXCLUDE",
            })
    panel = pd.DataFrame(rows)
    n_real = panel[panel["is_real_change"]].groupby("metric").size().to_dict()
    n_plac = panel[~panel["is_real_change"]].groupby("metric").size().to_dict()
    print(f"\n  [{label}] real HC-change events per metric: {n_real}")
    print(f"  [{label}] placebo (continuous staff) events per metric: {n_plac}\n")

    print(f"  {'metric':<32} {'|d|_real':>10} {'|d|_plac':>10} {'ratio':>7} "
          f"{'p_perm':>8} {'verdict':>25}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*25}")
    summary = []
    for met in metrics:
        real = panel[(panel["metric"] == met) & panel["is_real_change"]]["abs_delta"].values
        plac = panel[(panel["metric"] == met) & ~panel["is_real_change"]]["abs_delta"].values
        if len(real) == 0 or len(plac) == 0:
            print(f"  {met:<32} insufficient data"); continue
        r_mean, p_mean = real.mean(), plac.mean()
        ratio = r_mean / p_mean if p_mean > 0 else np.nan
        _, p_val = permutation_test(real, plac)
        verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
                   else "NO DETECTABLE SHIFT" if p_val >= 0.1 else "MARGINAL")
        summary.append({"metric": met, "n_real": len(real), "n_plac": len(plac),
                        "abs_d_real_mean": r_mean, "abs_d_plac_mean": p_mean,
                        "ratio": ratio, "p_perm": p_val, "verdict": verdict})
        print(f"  {met:<32} {r_mean:>10.4f} {p_mean:>10.4f} {ratio:>7.3f} "
              f"{p_val:>8.4f} {verdict:>25}")
    return panel, pd.DataFrame(summary)


def main():
    print(f"=== HC OFFSEASON CHANGE × DEFENSIVE OUTCOMES (K={K} window) ===\n")
    print("Building staff panel + classifying offseason events...")
    staff = build_staff_panel()  # HC + OC; we use HC_CHANGE_EXCLUDE flag
    classified = classify_offseason_events(staff)
    print(f"  event class dist: {classified['event_class'].value_counts().to_dict()}")
    # Filter: only need HC_CHANGE_EXCLUDE (treatment) or CONTINUOUS (placebo)
    print(f"  HC_CHANGE_EXCLUDE (offseason HC change): {(classified['event_class'] == 'HC_CHANGE_EXCLUDE').sum()}")
    print(f"  CONTINUOUS (placebo): {(classified['event_class'] == 'CONTINUOUS').sum()}")

    print("\nBuilding GENERIC defensive residuals (Layer 2c parallel)...")
    sr_def_generic = build_defensive_residuals()
    print(f"  rows: {len(sr_def_generic):,}")
    gen_metrics = ["opp_pass_rate_residual","opp_plays_residual","opp_rz_trips_residual"]
    print(f"  metrics: {gen_metrics}")
    gen_panel, gen_summary = run_test(sr_def_generic, classified, gen_metrics,
                                       label="GENERIC")

    print("\n\nBuilding RZ-FINE-GRAIN defensive residuals (Layer 2e parallel)...")
    sr_def_rz = build_dc_rz_residuals()
    print(f"  rows: {len(sr_def_rz):,}")
    rz_metrics = ["opp_rz_pass_rate_residual","opp_rz_sack_rate_residual","opp_rz_td_rate_residual"]
    rz_panel, rz_summary = run_test(sr_def_rz, classified, rz_metrics, label="RZ FINE-GRAIN")

    # Save
    pd.concat([gen_panel, rz_panel], ignore_index=True).to_csv(
        ROOT / "hc_offseason_defense_panel.csv", index=False)
    pd.concat([gen_summary, rz_summary], ignore_index=True).to_csv(
        ROOT / "hc_offseason_defense_summary.csv", index=False)
    print(f"\n  wrote summary + panel CSVs")

    print(f"\n=== Reference: prior tests for comparison ===")
    print(f"  Layer 2c (DC change × generic def):     opp_pass 0.96x p=0.731, opp_RZ 1.09x p=0.110")
    print(f"  Layer 2e (DC change × RZ fine-grain):   opp_rz_pass 1.01x p=0.446, opp_rz_sack 1.06x p=0.238, opp_rz_td 0.92x p=0.851")
    print(f"  Layer 2a (HC mid-season × pass_rate):   1.36x p=0.002")
    print(f"  Layer 2d (OC change × RZ shotgun):      1.34x p=0.0002")

    print(f"\n=== Interpretation matrix ===")
    print(f"  IF HC offseason shifts opp_rz_X significantly:")
    print(f"     -> HC owns defensive identity at fine-grain (interpretation 2)")
    print(f"     -> Two-locus story extends: HC owns macro O+D, OC owns RZ-fine-grain O,")
    print(f"        DC has no detected locus, defensive identity is HC-locked")
    print(f"  IF HC offseason null on opp_rz_X:")
    print(f"     -> Defensive RZ identity is personnel-driven (interpretation 1)")
    print(f"     -> Coordinator role is asymmetric AND defensive identity is non-coaching")


if __name__ == "__main__":
    main()
