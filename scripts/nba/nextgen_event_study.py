"""NextGen tracking event study — finer-grain schematic metrics for NBA HCs.

NBA Projections has per-game player-level tracking via
  data/parquet/player_tracking_game.parquet (2019-20 to 2024-25, 183k rows).

Aggregate to per-team-game, compute season-league-mean residuals, run the
same offseason HC event study used at the box-score level.

NextGen metrics (paper hypothesis for each):

  passes_per_game             - ball-movement philosophy (HC)
  touches_per_game            - ball-handling philosophy (HC)
  contested_fga_per_game      - shot-quality choice (HC offensive identity)
  contested_fga_rate          - contested / total FGA (offensive scheme)
  defended_at_rim_fga_per_game - DEFENSIVE interior scheme (drop coverage
                                 forces this up; switching forces it down)
  defended_at_rim_pct          - opp rim FG% (defensive shot-defense)

defended_at_rim_fga is the finer-grain analog of opp_3PT_rate:
  if opp_3PT_rate fired (perimeter scheme), does opp_rim_fga ALSO fire
  (interior scheme)? Together they'd characterize HC defensive scheme as
  a whole (perimeter + interior coverage philosophy).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "nba" / "inputs"   # staged per mode: scripts/build/stage_nba.py
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

K = 10
N_PERM = 5000
SEED = 999


def build_team_game_nextgen() -> pd.DataFrame:
    """Aggregate player_tracking_game to per-team-game NextGen metrics."""
    pg = pd.read_parquet(NBA_PROJ / "player_tracking_game.parquet")
    pg = pg[pg["season_type"] == "Regular Season"].copy()
    tg = (pg.groupby(["game_id", "season", "team_tricode"], as_index=False)
            .agg(passes=("passes", "sum"),
                 touches=("touches", "sum"),
                 contested_fgm=("contested_field_goals_made", "sum"),
                 contested_fga=("contested_field_goals_attempted", "sum"),
                 uncontested_fgm=("uncontested_field_goals_made", "sum"),
                 uncontested_fga=("uncontested_field_goals_attempted", "sum"),
                 defended_rim_fgm=("defended_at_rim_field_goals_made", "sum"),
                 defended_rim_fga=("defended_at_rim_field_goals_attempted", "sum")))
    tg = tg.rename(columns={"team_tricode": "team_abbr"})
    tg["total_fga"] = tg["contested_fga"] + tg["uncontested_fga"]
    tg["contested_fga_rate"] = tg["contested_fga"] / tg["total_fga"].clip(lower=1)
    tg["defended_rim_fg_pct"] = tg["defended_rim_fgm"] / tg["defended_rim_fga"].clip(lower=1)
    # Already per-game, no further division
    # Add per-season residuals
    metrics = ["passes", "touches", "contested_fga", "contested_fga_rate",
                "defended_rim_fga", "defended_rim_fg_pct"]
    for m in metrics:
        smean = tg.groupby("season")[m].transform("mean")
        tg[f"{m}_residual"] = tg[m] - smean
    # Game order per (team, season) — use game_date if available, else game_id sort
    # No game_date in player_tracking_game; use game_id which is chronological for nba_api
    tg = tg.sort_values(["season", "team_abbr", "game_id"])
    tg["game_in_season"] = tg.groupby(["season", "team_abbr"]).cumcount() + 1
    return tg


def detect_events():
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    off = flags[(flags["new_coach_this_season"] == True)
                  & (flags["mid_season_change"] == False)]
    cont = flags[(flags["new_coach_this_season"] == False)
                   & (flags["mid_season_change"] == False)]
    return off, cont


def compute_pre_post(resid, team, season_str, metric, k=K):
    sy = int(season_str.split("-")[0])
    prior_str = f"{sy - 1}-{str(sy % 100).zfill(2)}"
    prev = resid[(resid["team_abbr"] == team) & (resid["season"] == prior_str)].sort_values("game_in_season")
    curr = resid[(resid["team_abbr"] == team) & (resid["season"] == season_str)].sort_values("game_in_season")
    if len(prev) < k or len(curr) < k:
        return None
    return float(curr.iloc[:k][metric].mean() - prev.iloc[-k:][metric].mean())


def perm_test(real, plac, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = real.mean() - plac.mean()
    a = np.concatenate([real, plac])
    n_r = len(real)
    null = np.zeros(n)
    for i in range(n):
        s = rng.permutation(a)
        null[i] = s[:n_r].mean() - s[n_r:].mean()
    return obs, float((null >= obs).mean())


def run(metric, off, cont, resid):
    real_d, plac_d = [], []
    for _, ev in off.iterrows():
        d = compute_pre_post(resid, ev["team_abbr"], ev["season"], metric)
        if d is not None:
            real_d.append(abs(d))
    for _, ev in cont.iterrows():
        d = compute_pre_post(resid, ev["team_abbr"], ev["season"], metric)
        if d is not None:
            plac_d.append(abs(d))
    real = np.array(real_d); plac = np.array(plac_d)
    if len(real) == 0 or len(plac) == 0:
        return None
    ratio = real.mean() / plac.mean() if plac.mean() > 0 else np.nan
    _, p = perm_test(real, plac)
    verdict = ("STAFF SHIFT MOVES NEEDLE" if p < 0.05 and ratio > 1.1
               else "NO DETECTABLE SHIFT" if p >= 0.10 else "MARGINAL")
    return {"metric": metric, "n_real": len(real), "n_plac": len(plac),
            "real_mean": float(real.mean()), "plac_mean": float(plac.mean()),
            "ratio": float(ratio), "p_perm": p, "verdict": verdict}


def main():
    print(f"=== NBA NEXTGEN OFFSEASON HC EVENT STUDY ===")
    print(f"  K={K}, n_perm={N_PERM}")
    print(f"  Window: 2019-20 to 2024-25 (tracking-data overlap; smaller than box-score window)\n")
    resid = build_team_game_nextgen()
    print(f"  team-game rows: {len(resid):,}, seasons: {sorted(resid['season'].unique())}")

    off, cont = detect_events()
    print(f"  offseason events: {len(off)}, continuous-staff: {len(cont)}\n")

    metrics = [
        "passes_residual", "touches_residual",
        "contested_fga_residual", "contested_fga_rate_residual",
        "defended_rim_fga_residual", "defended_rim_fg_pct_residual",
    ]
    print(f"  {'metric':<34} {'n_real':>6} {'n_plac':>6} {'|d|_real':>9} "
          f"{'|d|_plac':>9} {'ratio':>7} {'p_perm':>8} verdict")
    print(f"  {'-'*34} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*8} {'-'*25}")
    rows = []
    for m in metrics:
        r = run(m, off, cont, resid)
        if r is None:
            print(f"  {m:<34} insufficient data"); continue
        print(f"  {m:<34} {r['n_real']:>6} {r['n_plac']:>6} {r['real_mean']:>9.4f} "
              f"{r['plac_mean']:>9.4f} {r['ratio']:>7.3f} {r['p_perm']:>8.4f} {r['verdict']}")
        rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "nextgen_offseason_hc_results.csv", index=False)
    print(f"\nwrote: {RESULTS / 'nextgen_offseason_hc_results.csv'}")

    print(f"\n  Cross-reference (own data layer, same offseason events):")
    print(f"    pace (own possessions): 1.20 p=0.048 (FIRES, but propensity-attenuated)")
    print(f"    opp 3PT rate:           1.21 p=0.040 (FIRES, survives propensity)")


if __name__ == "__main__":
    main()
