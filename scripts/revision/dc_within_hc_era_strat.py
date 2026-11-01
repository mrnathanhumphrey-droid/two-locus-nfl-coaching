"""Era stratification of DC-within-HC test: was v1's signal real or a noise spike?

Split the extended panel into era bins:
  2000-2008 (early, pre-Fangio-era)
  2009-2016 (mid)
  2017-2024 (modern, v1 sample)

If signal is structural (DC role strengthening), modern era should hit and earlier eras null.
If signal is noise spike, no era should reach significance independently and v1 was lucky.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dc_within_hc_event_study import OUTCOMES, permutation_test

DATA = Path(__file__).resolve().parents[2] / "data" / "nfl"
OUT = DATA / "sloan_revision_outputs"

ERAS = [(2001, 2008, "2001-2008"), (2009, 2016, "2009-2016"), (2017, 2024, "2017-2024")]


def main():
    events = pd.read_parquet(OUT / "dc_within_hc_events_panel_extended.parquet")
    print(f"Total events: {len(events)}", flush=True)

    print("\n" + "=" * 110, flush=True)
    print("ERA STRATIFICATION: DC-isolated events", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'cell':32s}  ", end="", flush=True)
    for _, _, label in ERAS:
        print(f"  {label:12s}  ", end="")
    print()
    rows = []   # replication package: the printed table, also saved at full precision (the only non-path change)
    for cell in OUTCOMES:
        cell_shift = f"{cell}_shift"
        print(f"  {cell:32s}  ", end="", flush=True)
        for lo, hi, label in ERAS:
            tr = events[(events["transition"] == "DC_isolated")
                          & (events["season"] >= lo) & (events["season"] <= hi)][cell_shift].dropna().values
            cont = events[(events["transition"] == "Continuous")
                            & (events["season"] >= lo) & (events["season"] <= hi)][cell_shift].dropna().values
            if len(tr) < 5 or len(cont) < 5:
                print(f"  n={len(tr):3d}    --      ", end="")
                continue
            obs_diff, p, n_t, n_c = permutation_test(tr, cont, B=1000, seed=42)
            t_abs = abs(tr.mean()); c_abs = abs(cont.mean())
            ratio = t_abs / c_abs if c_abs > 1e-9 else float("inf")
            sig = "*" if p < 0.05 else " "
            print(f"  n={n_t:3d} {ratio:5.2f}x{sig}p={p:.2f}", end="")
            rows.append({"cell": cell, "era": label, "n_treated": n_t, "n_control": n_c,
                         "treated_mean_abs": t_abs, "control_mean_abs": c_abs, "ratio": ratio, "p_perm": p})
        print()
    pd.DataFrame(rows).to_csv(OUT / "dc_within_hc_era_strat.csv", index=False)


if __name__ == "__main__":
    main()
