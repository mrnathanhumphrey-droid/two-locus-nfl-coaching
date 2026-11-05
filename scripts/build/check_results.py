"""Compare a run's outputs (results/<mode>/) with the expected ones (results/expected/<mode>/).

    python scripts/build/check_results.py submitted

Walks every subfolder (revision/, nba/). Every expected .csv, .json and .parquet must exist in the run and match
exactly: numbers to 1e-9 (all randomness in the analysis is seeded, the Bayesian fits included), text exactly
(except PROVENANCE keys such as the git commit a run came from).
Exit 1 on any difference, naming the file and the column or key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-9
PROVENANCE = {".git_commit"}   # JSON keys that record where a run came from, not a result


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> str:
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return f"rows {len(a)} vs expected {len(b)}"
    for c in a.columns:
        x, y = pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce")
        if x.notna().sum() >= 0.9 * max(a[c].notna().sum(), 1) and x.notna().any():
            x, y = x.astype(float), y.astype(float)
            if (x.isna() != y.isna()).any() or float((x - y).abs().max()) > TOL:
                return f"{c} differs (max |diff| {float((x - y).abs().max()):.3g})"
        elif not a[c].astype(str).reset_index(drop=True).equals(b[c].astype(str).reset_index(drop=True)):
            return f"{c} differs (text)"
    return ""


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def compare_json(a, b) -> str:
    la, lb = dict(leaves(a)), dict(leaves(b))
    if set(la) != set(lb):
        return f"keys differ: {sorted(set(la) ^ set(lb))[:4]}"
    for k, x in la.items():
        if k in PROVENANCE:
            continue
        y = lb[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if abs(float(x) - float(y)) > TOL:
                return f"{k} differs ({x} vs expected {y})"
        elif x != y:
            return f"{k} differs ({x!r} vs expected {y!r})"
    return ""


def compare(got: Path, want: Path) -> str:
    if not got.exists():
        return "missing"
    if want.suffix == ".csv":
        return compare_frames(pd.read_csv(got), pd.read_csv(want))
    if want.suffix == ".parquet":
        return compare_frames(pd.read_parquet(got), pd.read_parquet(want))
    return compare_json(json.loads(got.read_text()), json.loads(want.read_text()))


def main(mode: str) -> int:
    got, want = REPO / "results" / mode, REPO / "results" / "expected" / mode
    expected = sorted(p for p in want.rglob("*") if p.suffix in (".csv", ".json", ".parquet"))
    bad = 0
    for e in expected:
        rel = e.relative_to(want)
        why = compare(got / rel, e)
        print(("  OK    " if not why else "  FAIL  ") + rel.as_posix() + (f"  {why}" if why else ""))
        bad += bool(why)
    print(f"\n{mode}: {len(expected) - bad} of {len(expected)} outputs match the expected results")
    return 1 if bad or not expected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "submitted"))
