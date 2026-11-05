"""NBA Layer 2 (defensive analog) — does offseason HC change move defensive
schematic identity?

Mirrors NFL Layer 2c/e/f (DC + HC change × defensive metrics, all NULL).

Defensive metrics computed via self-join (opponent's offensive stats per game):
  opp_three_pt_rate_residual: opp 3PA/FGA minus season mean
  opp_eFG_residual:           opp (FGM + 0.5*FG3M)/FGA minus season mean
  opp_ast_rate_residual:      opp AST/FGM (ball-movement allowed) minus season mean
  opp_possessions_residual:   opp pace (game-level pace is shared, but include as sanity)

Design parallel to offensive event study: K=10 pre/post, offseason HC change
treated vs continuous-staff placebo, perm null B=5000.

Hypothesis: if NFL's structural defensive null is sport-general, NBA should
also return null. If NBA defense fires, defensive coaching has more leverage
in basketball (consistent with switching schemes, drop coverage, etc.).
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


def build_team_game_with_opp() -> pd.DataFrame:
    """Build team-game with own + opponent offensive stats (defensive analog
    metrics are just opponent's offensive stats from same game)."""
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    tg = (bs.groupby(["game_id", "season", "team_abbr", "game_date", "is_home"],
                       as_index=False)
            .agg(FGM=("FGM", "sum"), FGA=("FGA", "sum"),
                 FG3M=("FG3M", "sum"), FG3A=("FG3A", "sum"),
                 FTM=("FTM", "sum"), FTA=("FTA", "sum"),
                 OREB=("OREB", "sum"), DREB=("DREB", "sum"),
                 AST=("AST", "sum"), TOV=("TOV", "sum"),
                 PTS=("PTS", "sum")))
    # Self-join to get opponent stats
    opp = tg.rename(columns={"team_abbr": "opp_team",
                              "FGM": "opp_FGM", "FGA": "opp_FGA",
                              "FG3M": "opp_FG3M", "FG3A": "opp_FG3A",
                              "FTA": "opp_FTA", "FTM": "opp_FTM",
                              "OREB": "opp_OREB", "DREB": "opp_DREB",
                              "AST": "opp_AST", "TOV": "opp_TOV",
                              "PTS": "opp_PTS"})[
        ["game_id", "opp_team", "opp_FGM", "opp_FGA", "opp_FG3M", "opp_FG3A",
         "opp_FTA", "opp_FTM", "opp_OREB", "opp_AST", "opp_TOV", "opp_PTS"]
    ]
    merged = tg.merge(opp, on="game_id", how="left")
    # Filter to other-team rows (game_id has 2 teams; join gives 4 rows; keep only opp != self)
    merged = merged[merged["opp_team"] != merged["team_abbr"]].copy()

    # Defensive metrics
    merged["opp_three_pt_rate"] = merged["opp_FG3A"] / merged["opp_FGA"].clip(lower=1)
    merged["opp_eFG"] = (merged["opp_FGM"] + 0.5 * merged["opp_FG3M"]) / merged["opp_FGA"].clip(lower=1)
    merged["opp_ast_rate"] = merged["opp_AST"] / merged["opp_FGM"].clip(lower=1)
    merged["opp_possessions"] = merged["opp_FGA"] - merged["opp_OREB"] + merged["opp_TOV"] + 0.44 * merged["opp_FTA"]

    # Add residuals (per season league mean)
    for met in ["opp_three_pt_rate", "opp_eFG", "opp_ast_rate", "opp_possessions"]:
        season_mean = merged.groupby("season")[met].transform("mean")
        merged[f"{met}_residual"] = merged[met] - season_mean

    merged = merged.sort_values(["season", "team_abbr", "game_date"])
    merged["game_in_season"] = merged.groupby(["season", "team_abbr"]).cumcount() + 1
    return merged


def load_flags():
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    return flags


def compute_pre_post_delta(resid: pd.DataFrame, team: str, season_str: str,
                            metric: str, k: int = K) -> float | None:
    start_year = int(season_str.split("-")[0])
    prior_str = f"{start_year - 1}-{str(start_year % 100).zfill(2)}"
    prev = resid[(resid["team_abbr"] == team)
                  & (resid["season"] == prior_str)].sort_values("game_in_season")
    curr = resid[(resid["team_abbr"] == team)
                  & (resid["season"] == season_str)].sort_values("game_in_season")
    if len(prev) < k or len(curr) < k:
        return None
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    return float(post - pre)


def permutation_test(real_abs, plac_abs, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs_diff = real_abs.mean() - plac_abs.mean()
    all_vals = np.concatenate([real_abs, plac_abs])
    n_real = len(real_abs)
    null_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(all_vals)
        null_diffs[i] = shuffled[:n_real].mean() - shuffled[n_real:].mean()
    return float(obs_diff), float((null_diffs >= obs_diff).mean())


def run(metric, events, placebo, resid):
    real_d, plac_d = [], []
    for _, ev in events.iterrows():
        d = compute_pre_post_delta(resid, ev["team_abbr"], ev["season"], metric)
        if d is not None:
            real_d.append(abs(d))
    for _, ev in placebo.iterrows():
        d = compute_pre_post_delta(resid, ev["team_abbr"], ev["season"], metric)
        if d is not None:
            plac_d.append(abs(d))
    real = np.array(real_d); plac = np.array(plac_d)
    if len(real) == 0 or len(plac) == 0:
        return None
    ratio = real.mean() / plac.mean() if plac.mean() > 0 else np.nan
    _, p_val = permutation_test(real, plac)
    verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
               else "NO DETECTABLE SHIFT" if p_val >= 0.10
               else "MARGINAL")
    return {"metric": metric, "n_real": len(real), "n_plac": len(plac),
            "real_mean": float(real.mean()), "plac_mean": float(plac.mean()),
            "ratio": float(ratio), "p_perm": p_val, "verdict": verdict}


def main():
    print(f"=== NBA LAYER 2 (DEFENSIVE ANALOG, OFFSEASON HC CHANGE) ===")
    print(f"  K={K} games, n_perm={N_PERM}\n")
    print(f"Building team-game with opponent stats...")
    resid = build_team_game_with_opp()
    print(f"  rows: {len(resid):,}")

    flags = load_flags()
    off = flags[(flags["new_coach_this_season"] == True)
                  & (flags["mid_season_change"] == False)]
    cont = flags[(flags["new_coach_this_season"] == False)
                   & (flags["mid_season_change"] == False)]
    print(f"  offseason HC events: {len(off)}, continuous-staff placebos: {len(cont)}")

    metrics = ["opp_three_pt_rate_residual", "opp_eFG_residual",
                "opp_ast_rate_residual", "opp_possessions_residual"]
    print(f"\n  {'metric':<34} {'n_real':>6} {'n_plac':>6} {'|d|_real':>9} "
          f"{'|d|_plac':>9} {'ratio':>7} {'p_perm':>8} {'verdict':>25}")
    print(f"  {'-'*34} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*8} {'-'*25}")
    rows = []
    for met in metrics:
        r = run(met, off, cont, resid)
        if r is None:
            continue
        print(f"  {met:<34} {r['n_real']:>6} {r['n_plac']:>6} {r['real_mean']:>9.4f} "
              f"{r['plac_mean']:>9.4f} {r['ratio']:>7.3f} {r['p_perm']:>8.4f} "
              f"{r['verdict']:>25}")
        rows.append(r)
    out = RESULTS / "layer2_defensive_analog_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  wrote: {out}")

    print(f"\n  Cross-reference NFL defensive nulls (all 3 cells NULL or marginal):")
    print(f"    Layer 2c DC change x opp outcomes:  best opp_RZ_trips 1.09x p=0.110")
    print(f"    Layer 2e DC change x RZ defensive:  best opp_rz_pass 1.01x p=0.446")
    print(f"    Layer 2f HC offseason x defensive:  best opp_rz_pass 1.11x p=0.068")


if __name__ == "__main__":
    main()
