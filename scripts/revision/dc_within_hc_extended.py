"""DC-within-HC + HC-within-DC event study on EXTENDED defensive panel 2000-2024.

Same logic as dc_within_hc_event_study.py but uses defensive_outcome_panel_extended.parquet
(5x larger sample range). Expected DC-isolated events: ~150 (vs 31 in v1).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from dc_within_hc_event_study import (
    OUTCOMES, build_staff_wide, classify_transitions,
    compute_event_shifts, event_test,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "nfl"
OUT = DATA / "sloan_revision_outputs"


def main():
    print("=" * 80, flush=True)
    print("Sloan #2c — Extended-panel rerun (2000-2024)", flush=True)
    print("=" * 80, flush=True)

    wide = build_staff_wide()
    transitions = classify_transitions(wide)
    panel = pd.read_parquet(OUT / "defensive_outcome_panel_extended.parquet")
    print(f"Extended panel: {len(panel)} team-seasons, range {panel['season'].min()}-{panel['season'].max()}", flush=True)

    events = compute_event_shifts(transitions, panel)
    events_valid = events.dropna(subset=[f"{OUTCOMES[0]}_shift"]).copy()
    counts = events_valid["transition"].value_counts()
    print(f"\nValid event counts (extended):", flush=True)
    for k, v in counts.items():
        print(f"  {k}: {v}", flush=True)

    events_valid.to_parquet(OUT / "dc_within_hc_events_panel_extended.parquet")

    results = []
    for isolation in ["DC_isolated", "HC_isolated", "Both_change"]:
        for cell in OUTCOMES:
            r = event_test(isolation, events_valid, cell)
            results.append(r)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "dc_within_hc_results_extended.csv", index=False)

    print("\n" + "=" * 100, flush=True)
    print("EXTENDED RESULTS: ratio = |treated mean shift| / |continuous mean shift|", flush=True)
    print("=" * 100, flush=True)
    for isolation in ["DC_isolated", "HC_isolated", "Both_change"]:
        print(f"\n--- {isolation} ---", flush=True)
        sub = results_df[results_df["isolation"] == isolation].sort_values("ratio_treated_to_control", ascending=False)
        for _, row in sub.iterrows():
            verdict = " (SIG)" if row["p_perm"] < 0.05 and row["n_treated"] >= 10 else ""
            print(f"  {row['cell']:32s}  n={int(row['n_treated']):3d}  ratio={row['ratio_treated_to_control']:.2f}x  p={row['p_perm']:.3f}{verdict}", flush=True)

    # Side-by-side comparison vs v1 (2017-2024)
    print("\n" + "=" * 100, flush=True)
    print("v1 (2017-2024) vs v2 (2000-2024) comparison — DC-isolated cells", flush=True)
    print("=" * 100, flush=True)
    v1 = pd.read_csv(OUT / "dc_within_hc_results.csv")
    v1_dc = v1[v1["isolation"] == "DC_isolated"].set_index("cell")
    v2_dc = results_df[results_df["isolation"] == "DC_isolated"].set_index("cell")
    for cell in OUTCOMES:
        r1 = v1_dc.loc[cell] if cell in v1_dc.index else None
        r2 = v2_dc.loc[cell] if cell in v2_dc.index else None
        if r1 is not None and r2 is not None:
            print(f"  {cell:32s}  v1: n={int(r1['n_treated']):3d} ratio={r1['ratio_treated_to_control']:.2f}x p={r1['p_perm']:.3f}  |  v2: n={int(r2['n_treated']):3d} ratio={r2['ratio_treated_to_control']:.2f}x p={r2['p_perm']:.3f}", flush=True)


if __name__ == "__main__":
    main()
