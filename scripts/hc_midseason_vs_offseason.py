"""HC mid-season vs offseason event separation (paper natural-experiment cleanup).

Layer 2a's headline (HC change -> pass-rate shift 1.36x p=0.002) was computed
on the union of mid-season + offseason HC changes. This script separates the
two natural experiments and reports each cleanly:

  Mid-season HC change (n~49):
    - Interim coach takes over within-season
    - Often crisis-driven (firing of underperforming HC)
    - Less prep time; interim often retains existing scheme
    - Within-season design: pre = first k games (under prior HC), post = remaining
      games (under interim/new). k = change_after_games from PFR splits.

  Offseason HC change (n~160):
    - Deliberate hire with full offseason prep
    - New HC installs their schematic identity
    - Roster turnover possible but team continuity meaningful
    - Cross-season design: pre = last K=6 games of season N-1 (prior HC),
      post = first K=6 games of season N (new HC).

Hypothesis: offseason events should show LARGER absolute pass-rate shift than
mid-season events, because deliberate hires bring deeper schematic identity
change than interims do.

Same permutation-null methodology as Layer 2a: real |Δ| mean vs placebo
(continuous-staff team-seasons) mean via B=5000 permutation. SNR-discipline
gates retained.
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
    PFR_TO_NFL, LEGACY_REMAP, build_staff_panel, permutation_test, N_PERM,
)

K_OFFSEASON = 6
SEED = 42


def parse_midseason_events() -> pd.DataFrame:
    """PFR coaching_history.csv 'head_coach_games_split' -> mid-season events.
    Returns: team_pfr, team_nfl, season, change_after_games."""
    wide = pd.read_csv(PFR / "coaching_history.csv")
    splits = wide.dropna(subset=["head_coach_games_split"]).copy()
    pat = re.compile(r"\((\d+)-(\d+)(?:-(\d+))?\)\s+and\s+", re.IGNORECASE)
    rows = []
    for _, r in splits.iterrows():
        s = str(r["head_coach_games_split"])
        m = pat.search(s)
        if not m:
            continue
        w, l = int(m.group(1)), int(m.group(2))
        t = int(m.group(3)) if m.group(3) else 0
        change_after = w + l + t
        rows.append({
            "team_pfr": r["team"], "season": int(r["season"]),
            "change_after_games": change_after,
            "split_raw": s,
        })
    df = pd.DataFrame(rows)
    df["team_nfl"] = df["team_pfr"].map(PFR_TO_NFL)
    return df.dropna(subset=["team_nfl"])


def detect_offseason_hc_events(staff_panel: pd.DataFrame) -> pd.DataFrame:
    """team-seasons where HC[N] != HC[N-1] AND no mid-season change in N or N-1.
    Returns: team_nfl, season, HC_prev, HC_new."""
    s = staff_panel.copy().sort_values(["team_nfl", "season"]).reset_index(drop=True)
    s["HC_prev"] = s.groupby("team_nfl")["HC"].shift(1)
    s["season_prev"] = s.groupby("team_nfl")["season"].shift(1)
    s = s[(s["season"] - s["season_prev"]) == 1]
    s = s.dropna(subset=["HC", "HC_prev"])
    changes = s[s["HC"] != s["HC_prev"]].copy()

    # Exclude team-seasons that ALSO had a mid-season change (in season N)
    mid = parse_midseason_events()
    mid_keys = set(zip(mid["team_nfl"], mid["season"]))
    changes["is_clean_offseason"] = ~changes.apply(
        lambda r: (r["team_nfl"], r["season"]) in mid_keys, axis=1
    )
    clean = changes[changes["is_clean_offseason"]].copy()
    return clean[["team_nfl", "season", "HC_prev", "HC"]].rename(
        columns={"HC": "HC_new"}
    )


def detect_continuous_staff_offseason(staff_panel: pd.DataFrame) -> pd.DataFrame:
    """team-seasons where HC[N] == HC[N-1] AND OC[N] == OC[N-1] AND no mid-season
    change in N or N-1. Placebo pool for the offseason test."""
    s = staff_panel.copy().sort_values(["team_nfl", "season"]).reset_index(drop=True)
    s["HC_prev"] = s.groupby("team_nfl")["HC"].shift(1)
    s["OC_prev"] = s.groupby("team_nfl")["OC"].shift(1)
    s["season_prev"] = s.groupby("team_nfl")["season"].shift(1)
    s = s[(s["season"] - s["season_prev"]) == 1]
    s = s.dropna(subset=["HC", "HC_prev", "OC", "OC_prev"])
    cont = s[(s["HC"] == s["HC_prev"]) & (s["OC"] == s["OC_prev"])].copy()

    mid = parse_midseason_events()
    mid_keys = set(zip(mid["team_nfl"], mid["season"]))
    cont = cont[~cont.apply(lambda r: (r["team_nfl"], r["season"]) in mid_keys, axis=1)]
    return cont[["team_nfl", "season"]]


def load_state_residuals() -> pd.DataFrame:
    sr = pd.read_parquet(ROOT / "games" / "state_residuals.parquet")
    if "week" not in sr.columns:
        games = pd.read_parquet(ROOT / "games" / "games.parquet")
        sr = sr.merge(games[["game_id", "week"]], on="game_id", how="left")
    sr["team_canon"] = sr["posteam"].map(LEGACY_REMAP).fillna(sr["posteam"])
    return sr


def midseason_pre_post(sr: pd.DataFrame, team: str, season: int,
                       change_after_games: int, metric: str) -> float | None:
    g = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week").reset_index(drop=True)
    if len(g) < change_after_games + 2:
        return None
    pre = g.iloc[:change_after_games]
    post = g.iloc[change_after_games:]
    return float(post[metric].mean() - pre[metric].mean())


def offseason_pre_post(sr: pd.DataFrame, team: str, season: int,
                        metric: str, k: int = K_OFFSEASON) -> float | None:
    prev = sr[(sr["team_canon"] == team) & (sr["season"] == season - 1)].sort_values("week")
    curr = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week")
    if len(prev) < k or len(curr) < k:
        return None
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    return float(post - pre)


def midseason_placebo(sr: pd.DataFrame, real_keys: set, metric: str,
                       n_placebo_per_team_season: int = 1,
                       seed: int = 123) -> pd.DataFrame:
    """For each stable team-season (no HC change), pick fake change-week uniformly
    in [4, n_games - 4]. Return placebo deltas."""
    rng = np.random.default_rng(seed)
    stable = (sr.groupby(["team_canon", "season"]).size()
                .reset_index(name="n_games"))
    stable = stable[~stable.apply(
        lambda r: (r["team_canon"], r["season"]) in real_keys, axis=1)]
    stable = stable[stable["n_games"] >= 10]
    rows = []
    for _, srow in stable.iterrows():
        tm, ssn, n = srow["team_canon"], srow["season"], srow["n_games"]
        ts = sr[(sr["team_canon"] == tm) & (sr["season"] == ssn)].sort_values("week").reset_index(drop=True)
        k = int(rng.integers(4, n - 3))
        pre = ts.iloc[:k]
        post = ts.iloc[k:]
        rows.append({"team": tm, "season": ssn,
                     "delta": float(post[metric].mean() - pre[metric].mean())})
    return pd.DataFrame(rows)


def run_subset_test(real_deltas: list[float], placebo_deltas: list[float],
                     label: str, metric_name: str) -> dict:
    real = np.array([abs(d) for d in real_deltas if d is not None and not np.isnan(d)])
    plac = np.array([abs(d) for d in placebo_deltas if d is not None and not np.isnan(d)])
    if len(real) == 0 or len(plac) == 0:
        return {"label": label, "metric": metric_name, "n_real": len(real),
                "n_plac": len(plac), "real_mean": np.nan, "plac_mean": np.nan,
                "ratio": np.nan, "p_perm": np.nan, "verdict": "insufficient"}
    r_mean = float(real.mean())
    p_mean = float(plac.mean())
    ratio = r_mean / p_mean if p_mean > 0 else np.nan
    obs_diff, p_val = permutation_test(real, plac)
    verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
               else "NO DETECTABLE SHIFT" if p_val >= 0.10
               else "MARGINAL")
    return {"label": label, "metric": metric_name,
            "n_real": len(real), "n_plac": len(plac),
            "real_mean": r_mean, "plac_mean": p_mean,
            "ratio": ratio, "p_perm": float(p_val), "verdict": verdict}


def main():
    print(f"=== HC MID-SEASON vs OFFSEASON SEPARATION ===\n")

    staff = build_staff_panel()
    mid_events = parse_midseason_events()
    off_events = detect_offseason_hc_events(staff)
    cont_events = detect_continuous_staff_offseason(staff)

    print(f"mid-season HC events: {len(mid_events)}")
    print(f"offseason HC change events (clean, no mid-season in same season): {len(off_events)}")
    print(f"continuous-staff placebo events (offseason): {len(cont_events)}")

    sr = load_state_residuals()
    metrics = ["pass_rate_residual", "plays_residual", "red_zone_trips_residual"]

    results = []

    # --- MID-SEASON ---
    real_keys_mid = set(zip(mid_events["team_nfl"], mid_events["season"]))
    print(f"\n--- mid-season subset ---")
    for met in metrics:
        real_d = []
        for _, ev in mid_events.iterrows():
            d = midseason_pre_post(sr, ev["team_nfl"], ev["season"],
                                    ev["change_after_games"], met)
            if d is not None:
                real_d.append(d)
        plac_df = midseason_placebo(sr, real_keys_mid, met, seed=123)
        plac_d = plac_df["delta"].tolist()
        r = run_subset_test(real_d, plac_d, "mid_season", met)
        results.append(r)
        print(f"  {met:<28} n_real={r['n_real']:>3} n_plac={r['n_plac']:>4} "
              f"|d|_real={r['real_mean']:.4f} |d|_plac={r['plac_mean']:.4f} "
              f"ratio={r['ratio']:.3f} p={r['p_perm']:.4f} -> {r['verdict']}")

    # --- OFFSEASON ---
    print(f"\n--- offseason subset (K={K_OFFSEASON}) ---")
    for met in metrics:
        real_d = []
        for _, ev in off_events.iterrows():
            d = offseason_pre_post(sr, ev["team_nfl"], ev["season"], met, K_OFFSEASON)
            if d is not None:
                real_d.append(d)
        plac_d = []
        for _, ev in cont_events.iterrows():
            d = offseason_pre_post(sr, ev["team_nfl"], ev["season"], met, K_OFFSEASON)
            if d is not None:
                plac_d.append(d)
        r = run_subset_test(real_d, plac_d, "offseason", met)
        results.append(r)
        print(f"  {met:<28} n_real={r['n_real']:>3} n_plac={r['n_plac']:>4} "
              f"|d|_real={r['real_mean']:.4f} |d|_plac={r['plac_mean']:.4f} "
              f"ratio={r['ratio']:.3f} p={r['p_perm']:.4f} -> {r['verdict']}")

    # --- SIDE-BY-SIDE COMPARISON ---
    print(f"\n=== SIDE-BY-SIDE COMPARISON ===")
    res_df = pd.DataFrame(results)
    pivot = res_df.pivot(index="metric", columns="label",
                          values=["n_real", "real_mean", "ratio", "p_perm"])
    print(pivot.to_string())

    out_path = ROOT / "hc_midseason_vs_offseason_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
