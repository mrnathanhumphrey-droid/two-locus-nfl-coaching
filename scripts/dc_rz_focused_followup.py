"""DC RZ-focused follow-up — symmetric to OC RZ test.

OC RZ test found OC owns RZ shotgun-rate (1.34x, p=0.0002). The symmetric
question for DC: what RZ-specific defensive metric shifts at DC changes
under HC continuity?

DC schematic decisions in RZ:
  - Pressure scheme (rushers, blitz frequency)
  - Coverage shell (zone vs man)
  - Box count vs spread

Direct measurable proxies from nflverse pbp:
  - opp_rz_sack_rate    — sacks / RZ pass plays (pass-rush effectiveness; DC scheme)
  - opp_rz_pass_rate    — opp's RZ pass selection (partially DC-threat-driven)
  - opp_rz_td_rate      — RZ TD allowed (overall DC effectiveness; outcome)

Same design as DC-change event study main test (HC continuity, K=8 window).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_change_event_study import PFR_TO_NFL, LEGACY_REMAP, permutation_test, N_PERM
from dc_change_event_study import build_staff_panel_with_dc, classify_dc_events

ROOT = Path(__file__).resolve().parent.parent / "data"
K = 8


def build_dc_rz_residuals():
    """Per-game DEFENSIVE RZ metrics. For team T's defense, look at the row
    where posteam=opp_of_T and aggregate RZ stats from opp's perspective."""
    cols = ["game_id","play_id","season","week","posteam","play_type",
            "yardline_100","qb_kneel","qb_spike","two_point_attempt",
            "touchdown","sack","drive"]
    pbp = pd.read_parquet(ROOT / "games" / "pbp.parquet", columns=cols)
    pbp = pbp[(pbp["play_type"].isin(["pass","run"]))
              & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)
              & (pbp["two_point_attempt"] != 1)
              & pbp["posteam"].notna()].copy()
    rz = pbp[pbp["yardline_100"] <= 20].copy()
    rz["is_pass"] = (rz["play_type"] == "pass").astype(int)
    rz["is_sack"] = rz["sack"].fillna(0).astype(int)
    rz["is_td"] = rz["touchdown"].fillna(0).astype(int)

    # Per-team-game (posteam = offense): RZ rates
    tg = (rz.groupby(["game_id","season","week","posteam"]).agg(
        rz_plays=("play_id","count"),
        rz_pass_plays=("is_pass","sum"),
        rz_pass_rate=("is_pass","mean"),
        rz_sacks=("is_sack","sum"),
    ).reset_index())
    tg["rz_sack_rate"] = np.where(tg["rz_pass_plays"] > 0,
                                    tg["rz_sacks"] / tg["rz_pass_plays"], np.nan)
    # TD rate per drive
    drive_td = (rz.groupby(["game_id","season","week","posteam","drive"])["is_td"].max()
                  .reset_index().rename(columns={"is_td":"drive_had_td"}))
    rz_td = (drive_td.groupby(["game_id","season","week","posteam"]).agg(
        rz_trips=("drive","nunique"),
        rz_td_count=("drive_had_td","sum"),
    ).reset_index())
    rz_td["rz_td_rate"] = rz_td["rz_td_count"] / rz_td["rz_trips"]
    rz_panel = tg.merge(rz_td[["game_id","posteam","rz_td_rate"]],
                          on=["game_id","posteam"], how="left")
    rz_panel = rz_panel[rz_panel["rz_plays"] >= 3]

    # Flip to defense view via games table
    games = pd.read_parquet(ROOT / "games" / "games.parquet",
                              columns=["game_id","home_team","away_team"])
    rz_panel = rz_panel.merge(games, on="game_id", how="left")
    rz_panel["def_team"] = np.where(rz_panel["posteam"] == rz_panel["home_team"],
                                      rz_panel["away_team"], rz_panel["home_team"])
    # Rename to DEFENSE-allowed
    rz_panel = rz_panel.rename(columns={
        "rz_pass_rate": "opp_rz_pass_rate",
        "rz_sack_rate": "opp_rz_sack_rate",
        "rz_td_rate":   "opp_rz_td_rate",
    })

    # Residualize per season
    for met in ["opp_rz_pass_rate","opp_rz_sack_rate","opp_rz_td_rate"]:
        season_mean = rz_panel.groupby("season")[met].transform("mean")
        rz_panel[f"{met}_residual"] = rz_panel[met] - season_mean

    rz_panel["team_canon"] = rz_panel["def_team"].map(LEGACY_REMAP).fillna(rz_panel["def_team"])
    return rz_panel


def compute_pre_post(sr, team, season, metric, k=K):
    prev = sr[(sr["team_canon"] == team) & (sr["season"] == season-1)].sort_values("week")
    curr = sr[(sr["team_canon"] == team) & (sr["season"] == season)].sort_values("week")
    if len(prev) < k or len(curr) < k:
        return np.nan
    pre = prev.iloc[-k:][metric].mean()
    post = curr.iloc[:k][metric].mean()
    if np.isnan(pre) or np.isnan(post):
        return np.nan
    return float(post - pre)


def main():
    print(f"=== DC RZ-FOCUSED FOLLOW-UP (K={K} games per window) ===\n")
    staff = build_staff_panel_with_dc()
    classified = classify_dc_events(staff)
    print(f"DC event class dist: {classified['event_class'].value_counts().to_dict()}")

    print("\nBuilding RZ-defensive residuals...")
    rz = build_dc_rz_residuals()
    print(f"  RZ defensive team-game rows: {len(rz):,}")

    metrics = ["opp_rz_pass_rate_residual","opp_rz_sack_rate_residual","opp_rz_td_rate_residual"]
    rows = []
    for _, ev in classified.iterrows():
        if ev["event_class"] not in ("DC_CHANGE_HC_SAME","CONTINUOUS"):
            continue
        tm, ssn = ev["team_nfl"], int(ev["season"])
        for met in metrics:
            d = compute_pre_post(rz, tm, ssn, met)
            if np.isnan(d):
                continue
            rows.append({
                "team": tm, "season": ssn, "metric": met,
                "delta": d, "abs_delta": abs(d),
                "is_real_change": ev["event_class"] == "DC_CHANGE_HC_SAME",
            })
    panel = pd.DataFrame(rows)
    n_real = panel[panel["is_real_change"]].groupby("metric").size().to_dict()
    n_plac = panel[~panel["is_real_change"]].groupby("metric").size().to_dict()
    print(f"\n  real DC-change events per metric: {n_real}")
    print(f"  placebo (continuous staff) events per metric: {n_plac}\n")

    print(f"  {'metric':<32} {'|d|_real':>10} {'|d|_plac':>10} {'ratio':>7} "
          f"{'p_perm':>8} {'verdict':>25}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*25}")
    summary = []
    for met in metrics:
        real = panel[(panel["metric"] == met) & panel["is_real_change"]]["abs_delta"].values
        plac = panel[(panel["metric"] == met) & ~panel["is_real_change"]]["abs_delta"].values
        if len(real) == 0 or len(plac) == 0:
            print(f"  {met:<32} insufficient data"); continue
        r_mean, p_mean = real.mean(), plac.mean()
        ratio = r_mean / p_mean if p_mean > 0 else np.nan
        obs_diff, p_val = permutation_test(real, plac)
        verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
                   else "NO DETECTABLE SHIFT" if p_val >= 0.1 else "MARGINAL")
        summary.append({"metric": met, "n_real": len(real), "n_plac": len(plac),
                        "abs_d_real_mean": r_mean, "abs_d_plac_mean": p_mean,
                        "ratio": ratio, "p_perm": p_val, "verdict": verdict})
        print(f"  {met:<32} {r_mean:>10.4f} {p_mean:>10.4f} {ratio:>7.3f} "
              f"{p_val:>8.4f} {verdict:>25}")

    pd.DataFrame(summary).to_csv(ROOT / "dc_rz_focused_summary.csv", index=False)
    panel.to_csv(ROOT / "dc_rz_focused_panel.csv", index=False)
    print(f"\n  wrote summary + panel CSVs")

    print(f"\n=== Reference: OC RZ-focused (symmetric) ===")
    print(f"  rz_shotgun_rate_residual:  1.34x, p=0.0002  (STAFF SHIFT MOVES NEEDLE)")
    print(f"  rz_pass_rate_residual:     1.10x, p=0.075   (MARGINAL)")
    print(f"  rz_td_rate_residual:       0.99x, p=0.565   (NO DETECTABLE SHIFT)")


if __name__ == "__main__":
    main()
