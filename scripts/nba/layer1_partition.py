"""NBA Layer 1 partition F-test — does coaching vector explain season-level
schematic outcomes beyond team_id?

Mirrors NFL Layer 1. Expected NULL per NFL finding (season-aggregate is too
coarse for coaching to dominate over team identity).

Test: F-statistic of partition (coach OR team) on season-mean schematic
residuals. Permutation null B=1000 to bound F under random labeling.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "nba" / "inputs"   # staged per mode: scripts/build/stage_nba.py
RESULTS = ROOT / "data" / "results"

N_PERM = 1000


def f_stat(values, groups):
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if len(df) == 0: return np.nan, 0
    grand = df["v"].mean()
    gm = df.groupby("g")["v"].mean()
    gs = df.groupby("g")["v"].size()
    ng = len(gm); n = len(df)
    if ng < 2 or n - ng < 1: return np.nan, ng
    ss_b = float(((gm - grand) ** 2 * gs).sum())
    ss_w = float(sum(((df[df["g"] == g]["v"] - m) ** 2).sum() for g, m in gm.items()))
    if ss_w <= 0: return np.nan, ng
    F = (ss_b / (ng - 1)) / (ss_w / (n - ng))
    return F, ng


def build_season_aggregates():
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    tg = (bs.groupby(["game_id", "season", "team_abbr", "game_date"], as_index=False)
            .agg(FGM=("FGM", "sum"), FGA=("FGA", "sum"), FG3A=("FG3A", "sum"),
                 FTA=("FTA", "sum"), OREB=("OREB", "sum"), AST=("AST", "sum"),
                 TOV=("TOV", "sum")))
    tg["possessions"] = tg["FGA"] - tg["OREB"] + tg["TOV"] + 0.44 * tg["FTA"]
    tg["three_pt_rate"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    tg["ast_rate"] = tg["AST"] / tg["FGM"].clip(lower=1)
    # Opp 3PT rate via self-join
    opp = tg.rename(columns={"team_abbr": "opp_team",
                              "three_pt_rate": "opp_three_pt_rate"})[
        ["game_id", "opp_team", "opp_three_pt_rate"]]
    merged = tg.merge(opp, on="game_id", how="left")
    merged = merged[merged["opp_team"] != merged["team_abbr"]].copy()
    # Season-level aggregates per team
    season_agg = (merged.groupby(["team_abbr", "season"], as_index=False)
                          .agg(pace=("possessions", "mean"),
                                three_pt_rate=("three_pt_rate", "mean"),
                                ast_rate=("ast_rate", "mean"),
                                opp_three_pt_rate=("opp_three_pt_rate", "mean")))
    # Subtract season-league mean per metric to get residuals
    for met in ["pace", "three_pt_rate", "ast_rate", "opp_three_pt_rate"]:
        smean = season_agg.groupby("season")[met].transform("mean")
        season_agg[f"{met}_resid"] = season_agg[met] - smean
    # Join HC
    cts = pd.read_parquet(NBA_PROJ / "coach_team_season.parquet")
    cts = cts[["team_abbr", "season", "head_coach_name"]].rename(
        columns={"head_coach_name": "HC"})
    return season_agg.merge(cts, on=["team_abbr", "season"], how="left").dropna(subset=["HC"])


def main():
    print("=== NBA LAYER 1 (SEASON-AGGREGATE PARTITION F-TEST) ===\n")
    sa = build_season_aggregates()
    print(f"team-seasons with HC: {len(sa):,}")

    rng = np.random.default_rng(42)
    rows = []
    print(f"\n  {'metric':<24} {'class':<10} {'F_obs':>8} {'F_null_mean':>12} "
          f"{'F_null_p95':>12} {'p_one':>8} verdict")
    print(f"  {'-'*24} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*20}")
    for met in ["pace_resid", "three_pt_rate_resid", "ast_rate_resid", "opp_three_pt_rate_resid"]:
        y = sa[met].values
        for cls_name, cls_col in [("coach", "HC"), ("team", "team_abbr")]:
            g_obs = sa[cls_col].values
            F_obs, _ = f_stat(y, g_obs)
            null = []
            for _ in range(N_PERM):
                Fp, _ = f_stat(y, rng.permutation(g_obs))
                if not np.isnan(Fp):
                    null.append(Fp)
            null = np.array(null)
            p = float((null >= F_obs).mean())
            verdict = "SIGNAL" if p < 0.05 else "null"
            print(f"  {met:<24} {cls_name:<10} {F_obs:>8.3f} {null.mean():>12.3f} "
                  f"{np.percentile(null, 95):>12.3f} {p:>8.4f} {verdict}")
            rows.append({"metric": met, "class": cls_name, "F_obs": F_obs,
                          "F_null_mean": float(null.mean()),
                          "F_null_p95": float(np.percentile(null, 95)),
                          "p_one_sided": p, "verdict": verdict})

    pd.DataFrame(rows).to_csv(RESULTS / "layer1_partition_results.csv", index=False)
    print(f"\nwrote: {RESULTS / 'layer1_partition_results.csv'}")
    print(f"\nNFL Layer 1 reference: coaching vector 0/4 SIGNIFICANT at season aggregate;")
    print(f"team_id was strong on every metric. Expected NBA: similar pattern.")


if __name__ == "__main__":
    main()
