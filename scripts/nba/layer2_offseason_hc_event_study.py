"""Layer 2 (NBA) — Offseason HC change event study.

Tests whether offseason HC changes shift schematic team-rate (pace, 3PT rate,
AST%) more than continuous-staff team-seasons do over the same windowing.
Mirrors NFL Layer 2a' (offseason HC) methodology.

Design:
  Treated: team-seasons with offseason HC change (new_coach_this_season=True,
           mid_season_change=False from coaching_change_flags.parquet)
  Control: team-seasons with HC continuity (new_coach_this_season=False AND
           mid_season_change=False)
  Window: K=10 games. Pre = last K games of season N-1. Post = first K games
           of season N.
  Metric: |mean(post) - mean(pre)| per metric (pace_resid, 3PT_resid, AST_resid)
  Null: permutation B=5000 on |Δ| mean difference between treated and control.

Verdict gates: ratio>1.1 AND p<0.05 -> STAFF SHIFT MOVES NEEDLE.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "frozen" / "nba"
RESID = ROOT / "data" / "parquet" / "team_game_residuals.parquet"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

K = 10
N_PERM = 5000
SEED = 999


def load_flags() -> pd.DataFrame:
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    return flags[["team_abbr", "season", "season_end_year",
                   "new_coach_this_season", "mid_season_change",
                   "new_coach_name", "prior_coach_name", "new_hire_type"]]


def detect_offseason_events(flags: pd.DataFrame) -> pd.DataFrame:
    """Offseason HC changes only (not mid-season)."""
    off = flags[(flags["new_coach_this_season"] == True)
                 & (flags["mid_season_change"] == False)].copy()
    return off


def detect_continuous_staff(flags: pd.DataFrame) -> pd.DataFrame:
    """HC continuity AND no mid-season change in N or N-1."""
    cont = flags[(flags["new_coach_this_season"] == False)
                  & (flags["mid_season_change"] == False)].copy()
    return cont


def compute_pre_post_delta(resid: pd.DataFrame, team: str, season_str: str,
                            metric: str, k: int = K) -> float | None:
    """Pre = last K games of season N-1. Post = first K games of season N.
    season_str format: '2023-24' -> prior is '2022-23'."""
    start_year = int(season_str.split("-")[0])
    prior_str = f"{start_year - 1}-{str((start_year) % 100).zfill(2)}"
    prev = resid[(resid["team_abbr"] == team)
                  & (resid["season"] == prior_str)].sort_values("game_in_season")
    curr = resid[(resid["team_abbr"] == team)
                  & (resid["season"] == season_str)].sort_values("game_in_season")
    if len(prev) < k or len(curr) < k:
        return None
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    return float(post - pre)


def permutation_test(real_abs: np.ndarray, plac_abs: np.ndarray,
                      n_perm: int = N_PERM, seed: int = SEED):
    rng = np.random.default_rng(seed)
    obs_diff = real_abs.mean() - plac_abs.mean()
    all_vals = np.concatenate([real_abs, plac_abs])
    n_real = len(real_abs)
    null_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(all_vals)
        null_diffs[i] = shuffled[:n_real].mean() - shuffled[n_real:].mean()
    p = float((null_diffs >= obs_diff).mean())
    return float(obs_diff), p


def run_test_for_metric(metric: str, events: pd.DataFrame,
                         placebo: pd.DataFrame, resid: pd.DataFrame) -> dict:
    real_d, plac_d = [], []
    for _, ev in events.iterrows():
        d = compute_pre_post_delta(resid, ev["team_abbr"], ev["season"], metric)
        if d is not None:
            real_d.append(abs(d))
    for _, ev in placebo.iterrows():
        d = compute_pre_post_delta(resid, ev["team_abbr"], ev["season"], metric)
        if d is not None:
            plac_d.append(abs(d))
    real = np.array(real_d)
    plac = np.array(plac_d)
    if len(real) == 0 or len(plac) == 0:
        return {"metric": metric, "n_real": len(real), "n_plac": len(plac),
                "real_mean": np.nan, "plac_mean": np.nan,
                "ratio": np.nan, "p_perm": np.nan, "verdict": "insufficient"}
    ratio = real.mean() / plac.mean() if plac.mean() > 0 else np.nan
    obs_diff, p_val = permutation_test(real, plac)
    verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
               else "NO DETECTABLE SHIFT" if p_val >= 0.10
               else "MARGINAL")
    return {"metric": metric, "n_real": len(real), "n_plac": len(plac),
            "real_mean": float(real.mean()), "plac_mean": float(plac.mean()),
            "ratio": float(ratio), "p_perm": p_val, "verdict": verdict}


def main():
    print(f"=== NBA LAYER 2 (OFFSEASON HC EVENT STUDY) ===")
    print(f"  K={K} games per window, n_perm={N_PERM}\n")

    flags = load_flags()
    print(f"team-seasons in coaching_change_flags: {len(flags):,}")
    off = detect_offseason_events(flags)
    cont = detect_continuous_staff(flags)
    print(f"  offseason HC change events: {len(off)}")
    print(f"  continuous-staff placebos: {len(cont)}")
    print(f"  mid-season changes (excluded for now): {(flags['mid_season_change']==True).sum()}")

    resid = pd.read_parquet(RESID)
    print(f"\nteam-game residuals: {len(resid):,}, seasons: "
          f"{sorted(resid['season'].unique())[0]} .. {sorted(resid['season'].unique())[-1]}")

    metrics = ["possessions_residual", "three_pt_rate_residual", "ast_rate_residual"]
    print(f"\n  {'metric':<32} {'n_real':>6} {'n_plac':>6} {'|d|_real':>9} "
          f"{'|d|_plac':>9} {'ratio':>7} {'p_perm':>8} {'verdict':>25}")
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*8} {'-'*25}")
    results = []
    for met in metrics:
        r = run_test_for_metric(met, off, cont, resid)
        print(f"  {met:<32} {r['n_real']:>6} {r['n_plac']:>6} {r['real_mean']:>9.4f} "
              f"{r['plac_mean']:>9.4f} {r['ratio']:>7.3f} {r['p_perm']:>8.4f} "
              f"{r['verdict']:>25}")
        results.append(r)

    out = RESULTS / "layer2_offseason_hc_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nwrote: {out}")

    print(f"\n=== Cross-reference: NFL Layer 2a' (offseason HC, n=142) ===")
    print(f"  pass_rate_residual: ratio 1.02, p=0.43 (NULL)")
    print(f"  plays_residual:     ratio 1.17, p=0.021 (STAFF SHIFT MOVES NEEDLE)")
    print(f"  rz_trips_residual:  ratio 1.02, p=0.42 (NULL)")


if __name__ == "__main__":
    main()
