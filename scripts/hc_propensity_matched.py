"""Propensity-matched HC event design — causal rigor for the paper.

Addresses the "HC firings aren't random" referee concern. The naive Layer 2a
event-study compares firing team-seasons to stable team-seasons with a random
fake-split. If HC firings are correlated with team characteristics
(bad teams fire more; high-volatility teams fire more), the observed effect
could be confounded by team-quality differences, not the firing itself.

Two tests:
  (A) Mid-season pass_rate (Layer 2a headline, 1.36x p=0.002)
      - Treated: team-seasons with mid-season HC firing (n=49)
      - Control pool: team-seasons WITHOUT mid-season firing
      - Propensity: P(mid-season firing | prior_year_wins,
                         prior_year_pass_rate_residual_mean, hc_tenure_years,
                         season)
      - Match nearest-neighbor 1:1 within caliper 0.05
      - Apply matched control's data with the SAME split-week as treated
      - Compare |Δ pass_rate| treated vs matched control

  (B) Offseason plays/game (new finding, 1.17x p=0.021)
      - Treated: team-seasons with offseason HC change (n=164)
      - Control pool: team-seasons with HC AND OC continuity
      - Same propensity covariates, same matching
      - K=6 cross-season window
      - Compare |Δ plays/game|

If matched-comparison effect is similar to original placebo-comparison
effect, the signal is causal (not driven by team-quality selection).
"""
from __future__ import annotations
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"
PFR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pfr"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_change_event_study import (
    PFR_TO_NFL, LEGACY_REMAP, build_staff_panel, permutation_test,
)
from hc_midseason_vs_offseason import (
    parse_midseason_events, detect_offseason_hc_events,
    detect_continuous_staff_offseason, load_state_residuals,
    K_OFFSEASON,
)

SEED = 42
CALIPER = 0.05


def build_team_season_features(staff_panel: pd.DataFrame, sr: pd.DataFrame) -> pd.DataFrame:
    """For each (team, season), build pre-period features used as propensity
    covariates. Uses ONLY data from season-1 and earlier.

    Features:
      prior_wins: wins in season-1 (NaN for first observable season)
      prior_pass_rate_residual_mean: avg pass_rate_residual in season-1
      hc_tenure_years: years current HC has been with team at start of season
      season: year (controls for league trends)
    """
    # HC tenure: for each (team, season), count consecutive prior seasons same HC
    s = staff_panel.copy().sort_values(["team_nfl", "season"]).reset_index(drop=True)
    s["HC_prev"] = s.groupby("team_nfl")["HC"].shift(1)
    s["HC_same_as_prev"] = (s["HC"] == s["HC_prev"]).astype(int)

    # Tenure: cumulative count of consecutive same-HC seasons up to and including this one
    def cumcount_same(g):
        g = g.copy()
        tenure = []
        run = 0
        for same in g["HC_same_as_prev"]:
            run = run + 1 if same == 1 else 1
            tenure.append(run)
        g["hc_tenure_years"] = tenure
        return g
    s = s.groupby("team_nfl", group_keys=False).apply(cumcount_same)

    # prior season pass_rate_residual mean
    sr_team_season = (sr.groupby(["team_canon", "season"])["pass_rate_residual"]
                        .mean().reset_index()
                        .rename(columns={"pass_rate_residual": "pass_rate_residual_mean"}))
    sr_team_season["prior_season"] = sr_team_season["season"] + 1
    sr_team_season = sr_team_season.rename(
        columns={"pass_rate_residual_mean": "prior_pass_rate_residual_mean"})[
            ["team_canon", "prior_season", "prior_pass_rate_residual_mean"]
        ].rename(columns={"prior_season": "season"})

    # plays_residual mean prior
    sr_plays = (sr.groupby(["team_canon", "season"])["plays_residual"]
                  .mean().reset_index()
                  .rename(columns={"plays_residual": "plays_residual_mean"}))
    sr_plays["season"] = sr_plays["season"] + 1
    sr_plays = sr_plays.rename(columns={"plays_residual_mean": "prior_plays_residual_mean"})

    # team wins in season-1: aggregate from games or from coach_seasons.games
    # Use coach_seasons HC entries (which have games + wins fields per coach-team-season)
    cs = pd.read_parquet(ROOT / "coaching/coach_seasons.parquet")
    hc_wins = (cs[cs["role"] == "head_coach"]
                 .groupby(["team", "season"])
                 .agg(team_wins=("wins", "sum"))
                 .reset_index())
    hc_wins["team_nfl"] = hc_wins["team"].map(PFR_TO_NFL)
    hc_wins["next_season"] = hc_wins["season"] + 1
    prior_wins = hc_wins[["team_nfl", "next_season", "team_wins"]].rename(
        columns={"next_season": "season", "team_wins": "prior_wins"}
    )

    feats = s.merge(prior_wins, on=["team_nfl", "season"], how="left")
    feats = feats.merge(sr_team_season, left_on=["team_nfl", "season"],
                         right_on=["team_canon", "season"], how="left")
    feats = feats.drop(columns=["team_canon"])
    feats = feats.merge(sr_plays, left_on=["team_nfl", "season"],
                         right_on=["team_canon", "season"], how="left")
    feats = feats.drop(columns=["team_canon"])

    return feats[["team_nfl", "season", "HC", "HC_prev", "HC_same_as_prev",
                   "hc_tenure_years", "prior_wins",
                   "prior_pass_rate_residual_mean", "prior_plays_residual_mean"]]


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simple logistic regression via numpy + Newton's method (avoid sklearn dep
    to match the rest of the codebase style)."""
    # Standardize X
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    beta = np.zeros(Xs.shape[1])
    for _ in range(50):
        z = Xs @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = p * (1 - p)
        grad = Xs.T @ (y - p)
        H = -(Xs.T * W) @ Xs
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    z = Xs @ beta
    propensity = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return propensity


def nearest_match(treated: pd.DataFrame, control: pd.DataFrame,
                  propensity_col: str = "propensity", caliper: float = CALIPER,
                  seed: int = SEED) -> list[tuple[int, int]]:
    """1:1 nearest-neighbor without replacement within caliper.
    Returns list of (treated_idx, control_idx)."""
    rng = np.random.default_rng(seed)
    t = treated.reset_index(drop=False).rename(columns={"index": "t_idx"})
    c = control.reset_index(drop=False).rename(columns={"index": "c_idx"})
    # Shuffle treated order for fairness
    t_order = list(t.index)
    rng.shuffle(t_order)
    used_controls = set()
    matches = []
    for ti in t_order:
        t_prop = t.loc[ti, propensity_col]
        candidates = c[~c["c_idx"].isin(used_controls)]
        if len(candidates) == 0:
            continue
        diffs = (candidates[propensity_col] - t_prop).abs()
        if diffs.min() > caliper:
            continue
        best_local = diffs.idxmin()
        c_idx = c.loc[best_local, "c_idx"]
        matches.append((t.loc[ti, "t_idx"], c_idx))
        used_controls.add(c_idx)
    return matches


def midseason_abs_delta(sr: pd.DataFrame, team: str, season: int,
                         change_after_games: int, metric: str) -> float | None:
    g = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week").reset_index(drop=True)
    if len(g) < change_after_games + 2:
        return None
    pre = g.iloc[:change_after_games]
    post = g.iloc[change_after_games:]
    return abs(float(post[metric].mean() - pre[metric].mean()))


def offseason_abs_delta(sr: pd.DataFrame, team: str, season: int,
                         metric: str, k: int = K_OFFSEASON) -> float | None:
    prev = sr[(sr["team_canon"] == team) & (sr["season"] == season - 1)].sort_values("week")
    curr = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week")
    if len(prev) < k or len(curr) < k:
        return None
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    return abs(float(post - pre))


def run_midseason_propensity():
    print(f"\n=== A. MID-SEASON pass_rate PROPENSITY MATCH ===")
    staff = build_staff_panel()
    sr = load_state_residuals()
    mid = parse_midseason_events()
    feats = build_team_season_features(staff, sr)

    # Label: 1 if (team, season) had mid-season firing
    mid_keys = set(zip(mid["team_nfl"], mid["season"]))
    feats["midseason_fired"] = feats.apply(
        lambda r: 1 if (r["team_nfl"], r["season"]) in mid_keys else 0, axis=1)
    feats = feats.dropna(subset=["prior_wins", "prior_pass_rate_residual_mean", "hc_tenure_years"])
    print(f"  team-seasons with all covariates: {len(feats):,}")
    print(f"  mid-season firings (labeled): {feats['midseason_fired'].sum()}")

    X = feats[["prior_wins", "prior_pass_rate_residual_mean",
                "hc_tenure_years", "season"]].values.astype(float)
    y = feats["midseason_fired"].values.astype(float)
    feats["propensity"] = fit_logistic(X, y)
    print(f"  propensity mean (treated): {feats[feats['midseason_fired']==1]['propensity'].mean():.4f}")
    print(f"  propensity mean (control):  {feats[feats['midseason_fired']==0]['propensity'].mean():.4f}")

    treated = feats[feats["midseason_fired"] == 1].copy()
    control = feats[feats["midseason_fired"] == 0].copy()

    # Merge change_after_games onto treated so we know the split point per event
    treated = treated.merge(
        mid[["team_nfl", "season", "change_after_games"]],
        on=["team_nfl", "season"], how="left"
    )
    treated = treated.dropna(subset=["change_after_games"])

    matches = nearest_match(treated.reset_index(drop=True),
                              control.reset_index(drop=True),
                              caliper=CALIPER)
    print(f"  matched pairs (caliper={CALIPER}): {len(matches)}")

    # For each pair, compute |Δ pass_rate| for treated at real split, for control at SAME split
    t_deltas, c_deltas = [], []
    treated = treated.reset_index(drop=True)
    control = control.reset_index(drop=True)
    for ti, ci in matches:
        tr = treated.loc[ti]
        cr = control.loc[ci]
        k = int(tr["change_after_games"])
        t_d = midseason_abs_delta(sr, tr["team_nfl"], int(tr["season"]),
                                    k, "pass_rate_residual")
        c_d = midseason_abs_delta(sr, cr["team_nfl"], int(cr["season"]),
                                    k, "pass_rate_residual")
        if t_d is None or c_d is None:
            continue
        t_deltas.append(t_d)
        c_deltas.append(c_d)

    t_arr = np.array(t_deltas); c_arr = np.array(c_deltas)
    if len(t_arr) == 0:
        print("  no matched pairs with computable deltas")
        return None
    ratio = t_arr.mean() / c_arr.mean() if c_arr.mean() > 0 else np.nan
    _, p = permutation_test(t_arr, c_arr)
    print(f"  treated |d| mean: {t_arr.mean():.4f}  (n={len(t_arr)})")
    print(f"  matched control |d| mean: {c_arr.mean():.4f}")
    print(f"  ratio: {ratio:.3f}")
    print(f"  p_perm: {p:.4f}")
    print(f"  ORIGINAL placebo result: ratio 1.36 p=0.002")
    verdict = ("CAUSAL — survives propensity matching" if p < 0.05 and ratio > 1.1
               else "ATTENUATED but directional" if ratio > 1.05
               else "CONFOUNDED — does not survive matching")
    print(f"  VERDICT: {verdict}")
    return {
        "test": "mid_season_pass_rate", "n_matched": len(t_arr),
        "treated_mean": float(t_arr.mean()), "control_mean": float(c_arr.mean()),
        "ratio": float(ratio), "p_perm": float(p), "verdict": verdict,
    }


def run_offseason_propensity():
    print(f"\n=== B. OFFSEASON plays/game PROPENSITY MATCH ===")
    staff = build_staff_panel()
    sr = load_state_residuals()
    off = detect_offseason_hc_events(staff)
    cont = detect_continuous_staff_offseason(staff)
    feats = build_team_season_features(staff, sr)

    # Treated: offseason HC change team-seasons
    off_keys = set(zip(off["team_nfl"], off["season"]))
    # Control pool: HC AND OC continuity team-seasons
    cont_keys = set(zip(cont["team_nfl"], cont["season"]))
    feats["offseason_change"] = feats.apply(
        lambda r: 1 if (r["team_nfl"], r["season"]) in off_keys
        else (0 if (r["team_nfl"], r["season"]) in cont_keys else -1),
        axis=1
    )
    feats = feats[feats["offseason_change"] >= 0]
    feats = feats.dropna(subset=["prior_wins", "prior_plays_residual_mean",
                                   "hc_tenure_years"])
    print(f"  team-seasons in test universe: {len(feats):,}")
    print(f"  offseason changes (treated): {(feats['offseason_change']==1).sum()}")
    print(f"  continuous-staff (control pool): {(feats['offseason_change']==0).sum()}")

    # NOTE: hc_tenure_years EXCLUDED from offseason propensity model — it
    # deterministically predicts the change (treated teams have rookie HC
    # tenure=1 by definition; controls have tenure>=2). Including it causes
    # perfect separation and zero overlap. Offseason covariates: team quality
    # + scheme prior + season trend.
    X = feats[["prior_wins", "prior_plays_residual_mean",
                "season"]].values.astype(float)
    y = feats["offseason_change"].values.astype(float)
    feats["propensity"] = fit_logistic(X, y)
    print(f"  propensity mean (treated): {feats[feats['offseason_change']==1]['propensity'].mean():.4f}")
    print(f"  propensity mean (control):  {feats[feats['offseason_change']==0]['propensity'].mean():.4f}")

    treated = feats[feats["offseason_change"] == 1].reset_index(drop=True)
    control = feats[feats["offseason_change"] == 0].reset_index(drop=True)
    matches = nearest_match(treated, control, caliper=CALIPER)
    print(f"  matched pairs (caliper={CALIPER}): {len(matches)}")

    t_deltas, c_deltas = [], []
    for ti, ci in matches:
        tr = treated.loc[ti]
        cr = control.loc[ci]
        t_d = offseason_abs_delta(sr, tr["team_nfl"], int(tr["season"]),
                                    "plays_residual", K_OFFSEASON)
        c_d = offseason_abs_delta(sr, cr["team_nfl"], int(cr["season"]),
                                    "plays_residual", K_OFFSEASON)
        if t_d is None or c_d is None:
            continue
        t_deltas.append(t_d)
        c_deltas.append(c_d)

    t_arr = np.array(t_deltas); c_arr = np.array(c_deltas)
    if len(t_arr) == 0:
        print("  no matched pairs with computable deltas")
        return None
    ratio = t_arr.mean() / c_arr.mean() if c_arr.mean() > 0 else np.nan
    _, p = permutation_test(t_arr, c_arr)
    print(f"  treated |d| mean: {t_arr.mean():.4f}  (n={len(t_arr)})")
    print(f"  matched control |d| mean: {c_arr.mean():.4f}")
    print(f"  ratio: {ratio:.3f}")
    print(f"  p_perm: {p:.4f}")
    print(f"  ORIGINAL placebo result: ratio 1.17 p=0.021")
    verdict = ("CAUSAL — survives propensity matching" if p < 0.05 and ratio > 1.1
               else "ATTENUATED but directional" if ratio > 1.05
               else "CONFOUNDED — does not survive matching")
    print(f"  VERDICT: {verdict}")
    return {
        "test": "offseason_plays", "n_matched": len(t_arr),
        "treated_mean": float(t_arr.mean()), "control_mean": float(c_arr.mean()),
        "ratio": float(ratio), "p_perm": float(p), "verdict": verdict,
    }


def main():
    print(f"=== HC PROPENSITY-MATCHED EVENT DESIGN ===")
    print(f"Caliper: {CALIPER}, matching: 1:1 nearest-neighbor without replacement")
    print(f"Propensity covariates: prior_wins, prior_metric_residual_mean, hc_tenure_years, season")

    rows = []
    a = run_midseason_propensity()
    if a is not None: rows.append(a)
    b = run_offseason_propensity()
    if b is not None: rows.append(b)

    print(f"\n=== SUMMARY ===")
    for r in rows:
        print(f"  {r['test']:<22} n_matched={r['n_matched']:>3} "
              f"treated={r['treated_mean']:.4f} control={r['control_mean']:.4f} "
              f"ratio={r['ratio']:.3f} p={r['p_perm']:.4f} -> {r['verdict']}")

    out = ROOT / "hc_propensity_matched_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
