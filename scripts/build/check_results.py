"""Compare a run's outputs (results/<mode>/) with the expected ones (results/expected/<mode>/).

    python scripts/build/check_results.py submitted

Every numeric column must match exactly (all randomness in the analysis is seeded); text columns must match
too. Exit 1 on any difference, naming the file and column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def compare(a: pd.DataFrame, b: pd.DataFrame) -> str:
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return f"rows {len(a)} vs expected {len(b)}"
    for c in a.columns:
        x, y = pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce")
        if x.notna().sum() >= 0.9 * max(a[c].notna().sum(), 1) and x.notna().any():
            x, y = x.astype(float), y.astype(float)
            if (x.isna() != y.isna()).any() or float((x - y).abs().max()) > 1e-9:
                return f"{c} differs (max |diff| {float((x - y).abs().max()):.3g})"
        elif not a[c].astype(str).equals(b[c].astype(str)):
            return f"{c} differs (text)"
    return ""


def main(mode: str) -> int:
    got, want = REPO / "results" / mode, REPO / "results" / "expected" / mode
    expected = sorted(want.glob("*.csv"))
    bad = 0
    for e in expected:
        g = got / e.name
        why = "missing" if not g.exists() else compare(pd.read_csv(g), pd.read_csv(e))
        print(("  OK    " if not why else "  FAIL  ") + e.name + (f"  {why}" if why else ""))
        bad += bool(why)
    print(f"\n{mode}: {len(expected) - bad} of {len(expected)} outputs match the expected results")
    return 1 if bad or not expected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "submitted"))
