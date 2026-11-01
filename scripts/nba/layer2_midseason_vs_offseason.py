"""NBA Layer 2 mid-season vs offseason HC split.

Mirrors NFL hc_midseason_vs_offseason refinement. Tests whether the naive
NBA Layer 2 event-study effect lives in mid-season firings or offseason
hires (or both).

Mid-season: interim coach takes over mid-season. Split point = first HC's
  games played (from all_coaches_json wins+losses).
Offseason: new HC at start of season, no mid-season change.

Per-metric: pace, 3PT-rate, AST-rate (offensive); opp_3PT_rate, opp_AST_rate
  (defensive analog where naive test fired).
"""
from __future__ import annotations
from pathlib import Path
import ast
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "frozen" / "nba"
RESULTS = ROOT / "data" / "results"

K = 10
N_PERM = 5000
SEED = 999


def parse_midseason_events() -> pd.DataFrame:
    cts = pd.read_parquet(NBA_PROJ / "coach_team_season.parquet")
    mid = cts[cts["mid_season_change"] == True].copy()
    rows = []
    for _, r in mid.iterrows():
        raw = r["all_coaches_json"]
        if isinstance(raw, str):
            try:
                coaches = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
        else:
            coaches = raw
        if not coaches or len(coaches) < 2:
            continue
        first = coaches[0]
        wins = first.get("wins") or 0
        losses = first.get("losses") or 0
        change_after = int(wins) + int(losses)
        if change_after < 5 or change_after > 77:
            continue  # need pre + post windows
        rows.append({
            "team_abbr": r["team_abbr"], "season": r["season"],
            "change_after_games": change_after,
            "first_hc": first.get("head_coach_name"),
            "second_hc": coaches[1].get("head_coach_name"),
        })
    return pd.DataFrame(rows)


def detect_offseason_events_and_continuous() -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    off = flags[(flags["new_coach_this_season"] == True)
                 & (flags["mid_season_change"] == False)]
    cont = flags[(flags["new_coach_this_season"] == False)
                  & (flags["mid_season_change"] == False)]
    return off, cont


def build_residuals_full() -> pd.DataFrame:
    """Re-derive team-game residuals including opponent metrics (defensive
    analog). Concatenates own + opp metric residuals onto a single team-game
    panel."""
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    tg = (bs.groupby(["game_id", "season", "team_abbr", "game_date"],
                       as_index=False)
            .agg(FGM=("FGM", "sum"), FGA=("FGA", "sum"),
                 FG3M=("FG3M", "sum"), FG3A=("FG3A", "sum"),
                 FTA=("FTA", "sum"), FTM=("FTM", "sum"),
                 OREB=("OREB", "sum"), AST=("AST", "sum"),
                 TOV=("TOV", "sum"), PTS=("PTS", "sum")))
    tg["possessions"] = tg["FGA"] - tg["OREB"] + tg["TOV"] + 0.44 * tg["FTA"]
    tg["three_pt_rate"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    tg["ast_rate"] = tg["AST"] / tg["FGM"].clip(lower=1)
    opp = tg.rename(columns={"team_abbr": "opp_team",
                              "possessions": "opp_possessions",
                              "three_pt_rate": "opp_three_pt_rate",
                              "ast_rate": "opp_ast_rate"})[
        ["game_id", "opp_team", "opp_possessions",
         "opp_three_pt_rate", "opp_ast_rate"]
    ]
    merged = tg.merge(opp, on="game_id", how="left")
    merged = merged[merged["opp_team"] != merged["team_abbr"]].copy()
    for met in ["possessions", "three_pt_rate", "ast_rate",
                 "opp_three_pt_rate", "opp_ast_rate"]:
        smean = merged.groupby("season")[met].transform("mean")
        merged[f"{met}_residual"] = merged[met] - smean
    merged = merged.sort_values(["season", "team_abbr", "game_date"])
    merged["game_in_season"] = merged.groupby(["season", "team_abbr"]).cumcount() + 1
    return merged


def midseason_delta(resid, team, season, k, metric):
    g = resid[(resid["team_abbr"] == team)
              & (resid["season"] == season)].sort_values("game_in_season").reset_index(drop=True)
    if len(g) < k + 2:
        return None
    pre = g.iloc[:k]; post = g.iloc[k:]
    return float(post[metric].mean() - pre[metric].mean())


def offseason_delta(resid, team, season, k, metric):
    start_year = int(season.split("-")[0])
    prior = f"{start_year - 1}-{str(start_year % 100).zfill(2)}"
    prev = resid[(resid["team_abbr"] == team) & (resid["season"] == prior)].sort_values("game_in_season")
    curr = resid[(resid["team_abbr"] == team) & (resid["season"] == season)].sort_values("game_in_season")
    if len(prev) < k or len(curr) < k:
        return None
    return float(curr.iloc[:k][metric].mean() - prev.iloc[-k:][metric].mean())


def permutation_test(real, plac, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = real.mean() - plac.mean()
    all_v = np.concatenate([real, plac])
    n_r = len(real)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        s = rng.permutation(all_v)
        null[i] = s[:n_r].mean() - s[n_r:].mean()
    return obs, float((null >= obs).mean())


def midseason_placebo(resid, real_keys, metric, seed=123):
    rng = np.random.default_rng(seed)
    stable = resid.groupby(["team_abbr", "season"]).size().reset_index(name="n")
    stable = stable[~stable.apply(lambda r: (r["team_abbr"], r["season"]) in real_keys, axis=1)]
    stable = stable[stable["n"] >= 20]
    deltas = []
    for _, srow in stable.iterrows():
        tm, ssn, n = srow["team_abbr"], srow["season"], srow["n"]
        g = resid[(resid["team_abbr"] == tm) & (resid["season"] == ssn)].sort_values("game_in_season").reset_index(drop=True)
        k = int(rng.integers(8, n - 8))
        pre = g.iloc[:k]; post = g.iloc[k:]
        deltas.append(float(post[metric].mean() - pre[metric].mean()))
    return np.array([abs(d) for d in deltas if not np.isnan(d)])


def main():
    print(f"=== NBA LAYER 2 — MID-SEASON vs OFFSEASON SPLIT ===")
    print(f"  K={K} (offseason window only; mid-season uses change_after_games)\n")
    mid_ev = parse_midseason_events()
    off_ev, cont_ev = detect_offseason_events_and_continuous()
    print(f"  mid-season events: {len(mid_ev)}")
    print(f"  offseason events: {len(off_ev)}")
    print(f"  continuous-staff placebos: {len(cont_ev)}")

    resid = build_residuals_full()
    metrics = ["possessions_residual", "three_pt_rate_residual", "ast_rate_residual",
                "opp_three_pt_rate_residual", "opp_ast_rate_residual"]

    rows = []
    for met in metrics:
        # MID-SEASON
        real_d = []
        for _, ev in mid_ev.iterrows():
            d = midseason_delta(resid, ev["team_abbr"], ev["season"],
                                  int(ev["change_after_games"]), met)
            if d is not None:
                real_d.append(abs(d))
        real_arr = np.array(real_d)
        real_keys_mid = set(zip(mid_ev["team_abbr"], mid_ev["season"]))
        plac_arr = midseason_placebo(resid, real_keys_mid, met)
        if len(real_arr) > 0 and len(plac_arr) > 0:
            ratio = real_arr.mean() / plac_arr.mean() if plac_arr.mean() > 0 else np.nan
            _, p = permutation_test(real_arr, plac_arr)
            verdict = ("STAFF SHIFT MOVES NEEDLE" if p < 0.05 and ratio > 1.1
                       else "NO DETECTABLE SHIFT" if p >= 0.10 else "MARGINAL")
            rows.append({"window": "mid_season", "metric": met,
                         "n_real": len(real_arr), "n_plac": len(plac_arr),
                         "real_mean": float(real_arr.mean()),
                         "plac_mean": float(plac_arr.mean()),
                         "ratio": float(ratio), "p_perm": float(p),
                         "verdict": verdict})

        # OFFSEASON
        real_d = []
        for _, ev in off_ev.iterrows():
            d = offseason_delta(resid, ev["team_abbr"], ev["season"], K, met)
            if d is not None:
                real_d.append(abs(d))
        plac_d = []
        for _, ev in cont_ev.iterrows():
            d = offseason_delta(resid, ev["team_abbr"], ev["season"], K, met)
            if d is not None:
                plac_d.append(abs(d))
        real_arr = np.array(real_d); plac_arr = np.array(plac_d)
        if len(real_arr) > 0 and len(plac_arr) > 0:
            ratio = real_arr.mean() / plac_arr.mean() if plac_arr.mean() > 0 else np.nan
            _, p = permutation_test(real_arr, plac_arr)
            verdict = ("STAFF SHIFT MOVES NEEDLE" if p < 0.05 and ratio > 1.1
                       else "NO DETECTABLE SHIFT" if p >= 0.10 else "MARGINAL")
            rows.append({"window": "offseason", "metric": met,
                         "n_real": len(real_arr), "n_plac": len(plac_arr),
                         "real_mean": float(real_arr.mean()),
                         "plac_mean": float(plac_arr.mean()),
                         "ratio": float(ratio), "p_perm": float(p),
                         "verdict": verdict})

    print(f"\n{'window':<11} {'metric':<32} {'n_real':>6} {'real':>7} {'plac':>7} "
          f"{'ratio':>7} {'p':>7} verdict")
    print(f"{'-'*11} {'-'*32} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*25}")
    for r in rows:
        print(f"{r['window']:<11} {r['metric']:<32} {r['n_real']:>6} {r['real_mean']:>7.3f} "
              f"{r['plac_mean']:>7.3f} {r['ratio']:>7.3f} {r['p_perm']:>7.4f} {r['verdict']}")

    pd.DataFrame(rows).to_csv(RESULTS / "layer2_midseason_vs_offseason_results.csv", index=False)
    print(f"\nwrote: {RESULTS / 'layer2_midseason_vs_offseason_results.csv'}")


if __name__ == "__main__":
    main()
