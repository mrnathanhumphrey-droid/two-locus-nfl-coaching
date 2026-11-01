"""Every number in the extended abstract (docs/Sloan_Two_Locus_NFL_Abstract.pdf), next to the value this run
reproduced and the file it comes from.

    python scripts/build/check_abstract.py            # reads results/submitted/

A value AGREES when it rounds to the abstract's printed figure at the abstract's own precision. Exit 1 if a
reproduced value is missing or disagrees.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results" / "submitted"


def row(df, **where):
    m = pd.Series(True, index=df.index)
    for k, v in where.items():
        m &= df[k] == v
    hit = df[m]
    if len(hit) != 1:
        raise LookupError(f"{where}: {len(hit)} rows")
    return hit.iloc[0]


def j(name):
    return json.loads((RUN / "revision" / name).read_text())


def c(name):
    return pd.read_csv(RUN / name)


hm = c("hc_midseason_vs_offseason_results.csv")
rz = c("oc_rz_focused_summary.csv")
oc, hc = j("oc_rc_summary.json"), j("hc_rc_summary.json")
fam = pd.read_csv(RUN / "revision" / "oc_concept_family_results.csv")
era = pd.read_csv(RUN / "revision" / "dc_within_hc_era_strat.csv")
nba = pd.read_csv(RUN / "nba" / "propensity_with_roster_results.csv")

# (claim in the abstract, the abstract's figure, reproduced value, file)
CHECKS = [
    ("HC mid-season firing: pass-rate shift", "1.36", row(hm, label="mid_season", metric="pass_rate_residual")["ratio"],
     "hc_midseason_vs_offseason_results.csv"),
    ("  ... p", ".002", row(hm, label="mid_season", metric="pass_rate_residual")["p_perm"], ""),
    ("HC off-season change: pace shift", "1.17", row(hm, label="offseason", metric="plays_residual")["ratio"], ""),
    ("  ... p", ".021", row(hm, label="offseason", metric="plays_residual")["p_perm"], ""),
    ("OC change: red-zone shotgun-rate shift", "1.34", row(rz, metric="rz_shotgun_rate_residual")["ratio"],
     "oc_rz_focused_summary.csv"),
    ("  ... p", ".0002", row(rz, metric="rz_shotgun_rate_residual")["p_perm"], ""),
    ("OC portability slope (hierarchical)", "0.469", oc["Bayes_slope_mean"], "revision/oc_rc_summary.json"),
    ("  ... 95% CI low", "0.32", oc["Bayes_slope_q025"], ""),
    ("  ... 95% CI high", "0.62", oc["Bayes_slope_q975"], ""),
    ("HC portability slope (hierarchical)", "0.279", hc["Bayes_slope_mean"], "revision/hc_rc_summary.json"),
    ("  ... 95% CI low", "-0.01", hc["Bayes_slope_q025"], ""),
    ("  ... 95% CI high", "0.56", hc["Bayes_slope_q975"], ""),
    ("Concept families: lowest slope", "0.41", fam["ols_slope"].min(), "revision/oc_concept_family_results.csv"),
    ("Concept families: highest slope", "0.50", fam["ols_slope"].max(), ""),
    ("Concept families: count", "6", len(fam), ""),
    ("DC change, 2017-2024: opponent plays", "6.37", row(era, cell="opp_plays_residual", era="2017-2024")["ratio"],
     "revision/dc_within_hc_era_strat.csv"),
    ("  ... p", ".007", row(era, cell="opp_plays_residual", era="2017-2024")["p_perm"], ""),
    ("NBA: HC perimeter defense survives roster matching",
     "survives", row(nba, metric="opp_three_pt_rate_residual")["verdict"], "nba/propensity_with_roster_results.csv"),
]


def agrees(shown: str, value) -> bool:
    if isinstance(value, str):
        return shown.lower() in value.lower()
    digits = len(shown.split(".")[1]) if "." in shown else 0
    return round(float(value), digits) == float(shown)


bad = 0
print(f"{'claim':52s} {'abstract':>9s} {'reproduced':>12s}  agrees  file")
for claim, shown, value, f in CHECKS:
    ok = agrees(shown, value)
    bad += not ok
    v = value if isinstance(value, str) else f"{float(value):.4f}"
    print(f"{claim:52s} {shown:>9s} {v:>12s}  {'yes' if ok else 'NO ':3s}     {f}")
print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} abstract figures agree with the reproduced values")
sys.exit(1 if bad else 0)
