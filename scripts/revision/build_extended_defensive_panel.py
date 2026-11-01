"""Extend defensive_outcome_panel from 2017-2024 to 2000-2024.

Calls existing builder defensive_personnel_vs_coaching_decomp.build_defensive_outcome_panel
on the full historical year range. Saves to a new file so we don't overwrite the paper's
original §5.3 cohort artifact.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from defensive_personnel_vs_coaching_decomp import build_defensive_outcome_panel  # noqa

OUT = Path(__file__).resolve().parents[2] / "data" / "nfl" / "sloan_revision_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    years = list(range(2000, 2025))  # 2000-2024 inclusive
    print(f"Building defensive outcome panel for years {years[0]}-{years[-1]}...", flush=True)
    panel = build_defensive_outcome_panel(years)
    panel = panel.dropna(subset=["opp_pass_rate_residual", "opp_plays_residual"]).copy()
    print(f"  panel rows: {len(panel)}", flush=True)
    print(f"  unique team-seasons: {len(panel.drop_duplicates(subset=['team', 'season']))}", flush=True)
    print(f"  seasons: {sorted(panel['season'].unique())}", flush=True)
    print(f"  cols: {list(panel.columns)}", flush=True)
    print(f"  NaN counts:", flush=True)
    for c in panel.columns:
        if panel[c].dtype.kind in "fi":
            n_nan = panel[c].isna().sum()
            if n_nan > 0:
                print(f"    {c}: {n_nan} NaN", flush=True)

    out_path = OUT / "defensive_outcome_panel_extended.parquet"
    panel.to_parquet(out_path)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
