"""NBA propensity match — falsifier with roster-retention covariate.

User's critique: NBA HC effect on pace might be confounded with concurrent
roster turnover. HCs get hired to match the existing roster style; high-pace
teams hire fast coaches when their PG is fast.

Test design:
  For each team-season:
    Compute roster_retention = fraction of prior-year team minutes retained
                                 (sum minutes of top-N prior-year players
                                  that played for this team this season)
  Propensity model: P(offseason HC change | prior_wins, roster_retention,
                                              prior_pace_residual_mean, season)
  Match each HC change event 1:1 to a continuous-staff team-season within
  caliper 0.10.
  Re-run pace + opp_3PT_rate event study on matched sample.

If naive ratios (1.20 pace, 1.21 opp_3PT) survive matching: real HC effect.
If ratios attenuate to ~1.0: confounded with roster turnover (user's critique
correct). Either outcome is paper-grade.
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
TOP_N_PLAYERS = 8  # top-8 by minutes prior year used for retention calc
CALIPER = 0.10
N_PERM = 5000
SEED = 42


def build_player_team_season_minutes() -> pd.DataFrame:
    """Per (player, team, season) total regular-season minutes from box scores."""
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    pms = (bs.groupby(["nba_api_id", "team_abbr", "season"], as_index=False)
             .agg(min_total=("minutes", "sum"),
                  games_played=("game_id", "nunique")))
    return pms


def compute_roster_retention(pms: pd.DataFrame) -> pd.DataFrame:
    """For each (team, season), compute fraction of prior-year top-N team
    minutes that returned to the same team this season."""
    rows = []
    # For each (team, season), rank prior-year players by minutes for THAT team
    for (team, season), g in pms.groupby(["team_abbr", "season"]):
        # season as string like "2023-24"
        start_y = int(season.split("-")[0])
        prior_season = f"{start_y - 1}-{str(start_y % 100).zfill(2)}"
        prior_team = pms[(pms["team_abbr"] == team)
                          & (pms["season"] == prior_season)]
        if len(prior_team) < TOP_N_PLAYERS:
            rows.append({"team_abbr": team, "season": season,
                          "roster_retention": np.nan,
                          "prior_top_n_minutes_total": np.nan})
            continue
        top_n = prior_team.nlargest(TOP_N_PLAYERS, "min_total")
        top_ids = set(top_n["nba_api_id"])
        prior_min = top_n["min_total"].sum()
        # Minutes those players played for SAME team this season
        returning = g[g["nba_api_id"].isin(top_ids)]["min_total"].sum()
        retention = returning / prior_min if prior_min > 0 else np.nan
        rows.append({"team_abbr": team, "season": season,
                      "roster_retention": float(retention),
                      "prior_top_n_minutes_total": float(prior_min)})
    return pd.DataFrame(rows)


def build_team_game_residuals():
    """Re-derive own pace + opp_3pt_rate residuals (subset of full builder)."""
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    tg = (bs.groupby(["game_id", "season", "team_abbr", "game_date"],
                       as_index=False)
            .agg(FGM=("FGM", "sum"), FGA=("FGA", "sum"),
                 FG3A=("FG3A", "sum"), FTA=("FTA", "sum"),
                 OREB=("OREB", "sum"), TOV=("TOV", "sum")))
    tg["possessions"] = tg["FGA"] - tg["OREB"] + tg["TOV"] + 0.44 * tg["FTA"]
    tg["three_pt_rate"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    opp = tg.rename(columns={"team_abbr": "opp_team",
                              "three_pt_rate": "opp_three_pt_rate"})[
        ["game_id", "opp_team", "opp_three_pt_rate"]]
    merged = tg.merge(opp, on="game_id", how="left")
    merged = merged[merged["opp_team"] != merged["team_abbr"]].copy()
    for met in ["possessions", "opp_three_pt_rate"]:
        smean = merged.groupby("season")[met].transform("mean")
        merged[f"{met}_residual"] = merged[met] - smean
    merged = merged.sort_values(["season", "team_abbr", "game_date"])
    merged["game_in_season"] = merged.groupby(["season", "team_abbr"]).cumcount() + 1
    return merged


def fit_logistic(X, y):
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    beta = np.zeros(Xs.shape[1])
    for _ in range(50):
        z = np.clip(Xs @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
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
    z = np.clip(Xs @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def nearest_match(treated, control, prop_col="propensity", caliper=CALIPER, seed=SEED):
    rng = np.random.default_rng(seed)
    t = treated.reset_index(drop=False).rename(columns={"index": "t_idx"})
    c = control.reset_index(drop=False).rename(columns={"index": "c_idx"})
    t_order = list(t.index); rng.shuffle(t_order)
    used = set(); matches = []
    for ti in t_order:
        tp = t.loc[ti, prop_col]
        cands = c[~c["c_idx"].isin(used)]
        if len(cands) == 0: continue
        diffs = (cands[prop_col] - tp).abs()
        if diffs.min() > caliper: continue
        ci_local = diffs.idxmin()
        cidx = c.loc[ci_local, "c_idx"]
        matches.append((t.loc[ti, "t_idx"], cidx))
        used.add(cidx)
    return matches


def offseason_delta(resid, team, season, k, metric):
    start_y = int(season.split("-")[0])
    prior = f"{start_y - 1}-{str(start_y % 100).zfill(2)}"
    prev = resid[(resid["team_abbr"] == team) & (resid["season"] == prior)].sort_values("game_in_season")
    curr = resid[(resid["team_abbr"] == team) & (resid["season"] == season)].sort_values("game_in_season")
    if len(prev) < k or len(curr) < k:
        return None
    return float(curr.iloc[:k][metric].mean() - prev.iloc[-k:][metric].mean())


def permutation_test(r, p, n=N_PERM, seed=999):
    rng = np.random.default_rng(seed)
    obs = r.mean() - p.mean()
    a = np.concatenate([r, p]); n_r = len(r)
    null = np.zeros(n)
    for i in range(n):
        s = rng.permutation(a)
        null[i] = s[:n_r].mean() - s[n_r:].mean()
    return obs, float((null >= obs).mean())


def main():
    print("=== NBA PROPENSITY MATCH WITH ROSTER RETENTION ===\n")
    print(f"Building player-team-season minutes...")
    pms = build_player_team_season_minutes()
    print(f"  (player, team, season) rows: {len(pms):,}")

    print(f"\nComputing roster retention (top-{TOP_N_PLAYERS} prior-year players)...")
    retention = compute_roster_retention(pms)
    print(f"  team-seasons with retention: {len(retention):,}, "
          f"mean retention: {retention['roster_retention'].mean():.3f}")

    # Build feature table
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    flags = flags[["team_abbr", "season", "new_coach_this_season", "mid_season_change"]]
    # Prior wins
    cts = pd.read_parquet(NBA_PROJ / "coach_team_season.parquet")
    cts_wins = cts.groupby(["team_abbr", "season"], as_index=False)["of_record_wins"].sum()
    # season "2014-15" -> next "2015-16"; use start_year + 1 and (start_year + 2) % 100 for end
    def _next_season(s):
        sy = int(s.split("-")[0])
        ny = sy + 1
        return f"{ny}-{str((ny + 1) % 100).zfill(2)}"
    cts_wins["next_season"] = cts_wins["season"].apply(_next_season)
    prior_wins = cts_wins[["team_abbr", "next_season", "of_record_wins"]].rename(
        columns={"next_season": "season", "of_record_wins": "prior_wins"})

    feats = flags.merge(retention, on=["team_abbr", "season"], how="left")
    feats = feats.merge(prior_wins, on=["team_abbr", "season"], how="left")
    # Restrict to offseason-change or continuous-staff (drop mid-season)
    feats = feats[feats["mid_season_change"] == False].copy()
    feats["treated"] = feats["new_coach_this_season"].astype(int)
    # Keep only complete-feature rows
    feats = feats.dropna(subset=["roster_retention", "prior_wins"])
    feats["season_int"] = feats["season"].apply(lambda s: int(s.split("-")[0]))

    print(f"\nfeats universe: {len(feats):,} team-seasons "
          f"({(feats['treated']==1).sum()} treated, "
          f"{(feats['treated']==0).sum()} control)")
    print(f"  treated retention mean: {feats[feats['treated']==1]['roster_retention'].mean():.3f}")
    print(f"  control retention mean: {feats[feats['treated']==0]['roster_retention'].mean():.3f}")
    print(f"  treated prior_wins mean: {feats[feats['treated']==1]['prior_wins'].mean():.1f}")
    print(f"  control prior_wins mean: {feats[feats['treated']==0]['prior_wins'].mean():.1f}")

    # Propensity model
    X = feats[["roster_retention", "prior_wins", "season_int"]].values.astype(float)
    feats["propensity"] = fit_logistic(X, feats["treated"].values.astype(float))
    print(f"  propensity (treated) mean: {feats[feats['treated']==1]['propensity'].mean():.4f}")
    print(f"  propensity (control) mean: {feats[feats['treated']==0]['propensity'].mean():.4f}")

    treated = feats[feats["treated"] == 1].reset_index(drop=True)
    control = feats[feats["treated"] == 0].reset_index(drop=True)
    matches = nearest_match(treated, control, caliper=CALIPER)
    print(f"  matched pairs (caliper={CALIPER}): {len(matches)}")

    # Residuals + offseason deltas on matched pairs
    print(f"\nBuilding team-game residuals...")
    resid = build_team_game_residuals()

    results_rows = []   # replication package: the printed results, also saved at full precision (the only non-path change)
    for metric, naive_ratio, naive_p in [("possessions_residual", 1.20, 0.048),
                                            ("opp_three_pt_rate_residual", 1.21, 0.040)]:
        t_d, c_d = [], []
        for ti, ci in matches:
            tr = treated.loc[ti]; cr = control.loc[ci]
            td = offseason_delta(resid, tr["team_abbr"], tr["season"], K, metric)
            cd = offseason_delta(resid, cr["team_abbr"], cr["season"], K, metric)
            if td is None or cd is None:
                continue
            t_d.append(abs(td)); c_d.append(abs(cd))
        t_arr = np.array(t_d); c_arr = np.array(c_d)
        if len(t_arr) == 0:
            print(f"  {metric}: no matched pairs with computable deltas")
            continue
        ratio = t_arr.mean() / c_arr.mean() if c_arr.mean() > 0 else np.nan
        _, p = permutation_test(t_arr, c_arr)
        verdict = ("SURVIVES MATCHING" if p < 0.05 and ratio > 1.1
                   else "ATTENUATED but directional" if ratio > 1.05
                   else "CONFOUNDED (drops to <=1.05)")
        print(f"\n  {metric}:")
        print(f"    n_matched={len(t_arr)}  treated={t_arr.mean():.4f}  control={c_arr.mean():.4f}")
        print(f"    matched ratio={ratio:.3f}  p_perm={p:.4f}")
        print(f"    naive ratio={naive_ratio:.2f} (p={naive_p:.3f})  ->  matched: {verdict}")
        results_rows.append({"metric": metric, "n_matched": len(t_arr), "treated_mean": t_arr.mean(), "control_mean": c_arr.mean(), "matched_ratio": ratio, "p_perm": p, "verdict": verdict})
    RESULTS_DIR = ROOT / "data" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_rows).to_csv(RESULTS_DIR / "propensity_with_roster_results.csv", index=False)


if __name__ == "__main__":
    main()
