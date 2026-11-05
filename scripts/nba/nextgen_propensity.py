"""Falsify the NextGen defensive findings under roster-retention propensity match.

defended_rim_fga 1.60x p=0.005 and defended_rim_fg_pct 1.32x p=0.035 fired in
the naive event study. Both could be heavily personnel-driven (rim protector
identity travels with the player, not the coach). This is the user's "NBA is
hyper-individualized" critique applied to interior defense.

Test design (same as propensity_with_roster.py):
  - Propensity model: P(offseason HC change | roster_retention, prior_wins, season)
  - 1:1 nearest-neighbor match, caliper 0.10
  - Re-run event study on matched pairs for defended_rim_fga + fg_pct

If signal survives matching: real coach scheme.
If signal attenuates to null: confounded with rim-protector turnover.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "nba" / "inputs"   # staged per mode: scripts/build/stage_nba.py
RESULTS = ROOT / "data" / "results"

K = 10
TOP_N = 8
CALIPER = 0.10
N_PERM = 5000
SEED = 42


def build_nextgen_residuals():
    pg = pd.read_parquet(NBA_PROJ / "player_tracking_game.parquet")
    pg = pg[pg["season_type"] == "Regular Season"].copy()
    tg = (pg.groupby(["game_id", "season", "team_tricode"], as_index=False)
            .agg(defended_rim_fgm=("defended_at_rim_field_goals_made", "sum"),
                 defended_rim_fga=("defended_at_rim_field_goals_attempted", "sum")))
    tg = tg.rename(columns={"team_tricode": "team_abbr"})
    tg["defended_rim_fg_pct"] = tg["defended_rim_fgm"] / tg["defended_rim_fga"].clip(lower=1)
    for m in ["defended_rim_fga", "defended_rim_fg_pct"]:
        smean = tg.groupby("season")[m].transform("mean")
        tg[f"{m}_residual"] = tg[m] - smean
    tg = tg.sort_values(["season", "team_abbr", "game_id"])
    tg["game_in_season"] = tg.groupby(["season", "team_abbr"]).cumcount() + 1
    return tg


def build_roster_retention():
    """Same approach as propensity_with_roster: top-N prior-year minute share retained."""
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    pms = (bs.groupby(["nba_api_id", "team_abbr", "season"], as_index=False)
             .agg(min_total=("minutes", "sum")))
    rows = []
    for (team, season), g in pms.groupby(["team_abbr", "season"]):
        sy = int(season.split("-")[0])
        prior = f"{sy - 1}-{str(sy % 100).zfill(2)}"
        prior_team = pms[(pms["team_abbr"] == team) & (pms["season"] == prior)]
        if len(prior_team) < TOP_N:
            rows.append({"team_abbr": team, "season": season,
                          "roster_retention": np.nan})
            continue
        top_n = prior_team.nlargest(TOP_N, "min_total")
        top_ids = set(top_n["nba_api_id"])
        prior_min = top_n["min_total"].sum()
        returning = g[g["nba_api_id"].isin(top_ids)]["min_total"].sum()
        rows.append({"team_abbr": team, "season": season,
                      "roster_retention": float(returning / prior_min) if prior_min > 0 else np.nan})
    return pd.DataFrame(rows)


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
        try: step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError: break
        beta -= step
        if np.max(np.abs(step)) < 1e-8: break
    z = np.clip(Xs @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def match(treated, control, prop_col, caliper, seed=SEED):
    rng = np.random.default_rng(seed)
    t = treated.reset_index(drop=False).rename(columns={"index": "t_idx"})
    c = control.reset_index(drop=False).rename(columns={"index": "c_idx"})
    order = list(t.index); rng.shuffle(order)
    used = set(); matches = []
    for ti in order:
        tp = t.loc[ti, prop_col]
        cand = c[~c["c_idx"].isin(used)]
        if len(cand) == 0: continue
        diffs = (cand[prop_col] - tp).abs()
        if diffs.min() > caliper: continue
        ci_local = diffs.idxmin()
        cidx = c.loc[ci_local, "c_idx"]
        matches.append((t.loc[ti, "t_idx"], cidx))
        used.add(cidx)
    return matches


def offseason_delta(resid, team, season, k, metric):
    sy = int(season.split("-")[0])
    prior = f"{sy - 1}-{str(sy % 100).zfill(2)}"
    prev = resid[(resid["team_abbr"] == team) & (resid["season"] == prior)].sort_values("game_in_season")
    curr = resid[(resid["team_abbr"] == team) & (resid["season"] == season)].sort_values("game_in_season")
    if len(prev) < k or len(curr) < k: return None
    return float(curr.iloc[:k][metric].mean() - prev.iloc[-k:][metric].mean())


def perm_test(r, p, n=N_PERM, seed=999):
    rng = np.random.default_rng(seed)
    obs = r.mean() - p.mean()
    a = np.concatenate([r, p]); n_r = len(r)
    null = np.zeros(n)
    for i in range(n):
        s = rng.permutation(a)
        null[i] = s[:n_r].mean() - s[n_r:].mean()
    return obs, float((null >= obs).mean())


def main():
    print("=== NEXTGEN PROPENSITY MATCH (roster_retention covariate) ===\n")
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    flags = flags[["team_abbr", "season", "new_coach_this_season", "mid_season_change"]]
    cts = pd.read_parquet(NBA_PROJ / "coach_team_season.parquet")
    cts_wins = cts.groupby(["team_abbr", "season"], as_index=False)["of_record_wins"].sum()
    def _next(s):
        sy = int(s.split("-")[0]) + 1
        return f"{sy}-{str((sy + 1) % 100).zfill(2)}"
    cts_wins["season"] = cts_wins["season"].apply(_next)
    prior_wins = cts_wins.rename(columns={"of_record_wins": "prior_wins"})

    retention = build_roster_retention()
    feats = (flags.merge(retention, on=["team_abbr", "season"], how="left")
                  .merge(prior_wins, on=["team_abbr", "season"], how="left"))
    feats = feats[feats["mid_season_change"] == False].copy()
    feats = feats.dropna(subset=["roster_retention", "prior_wins"])
    feats["treated"] = feats["new_coach_this_season"].astype(int)
    feats["season_int"] = feats["season"].apply(lambda s: int(s.split("-")[0]))

    # Restrict to tracking-data seasons (2019-20 onward)
    feats = feats[feats["season_int"] >= 2019].copy()
    print(f"feats (tracking-overlap window): {len(feats):,} team-seasons "
          f"({(feats['treated']==1).sum()} treated, {(feats['treated']==0).sum()} control)")

    X = feats[["roster_retention", "prior_wins", "season_int"]].values.astype(float)
    feats["propensity"] = fit_logistic(X, feats["treated"].values.astype(float))

    treated = feats[feats["treated"] == 1].reset_index(drop=True)
    control = feats[feats["treated"] == 0].reset_index(drop=True)
    matches = match(treated, control, "propensity", CALIPER)
    print(f"matched pairs (caliper={CALIPER}): {len(matches)}\n")

    print(f"Building NextGen residuals...")
    resid = build_nextgen_residuals()

    naive_results = {
        "defended_rim_fga_residual": (1.595, 0.0048),
        "defended_rim_fg_pct_residual": (1.319, 0.035),
    }
    results_rows = []   # replication package: the printed results, also saved at full precision (the only non-path change)
    for metric, (naive_r, naive_p) in naive_results.items():
        t_d, c_d = [], []
        for ti, ci in matches:
            tr = treated.loc[ti]; cr = control.loc[ci]
            td = offseason_delta(resid, tr["team_abbr"], tr["season"], K, metric)
            cd = offseason_delta(resid, cr["team_abbr"], cr["season"], K, metric)
            if td is None or cd is None: continue
            t_d.append(abs(td)); c_d.append(abs(cd))
        t = np.array(t_d); c = np.array(c_d)
        if len(t) == 0:
            print(f"  {metric}: no matched pairs"); continue
        ratio = t.mean() / c.mean() if c.mean() > 0 else np.nan
        _, p = perm_test(t, c)
        verdict = ("SURVIVES MATCHING" if p < 0.05 and ratio > 1.1
                   else "ATTENUATED" if ratio > 1.05
                   else "CONFOUNDED")
        print(f"  {metric}:")
        print(f"    n_matched={len(t)}  treated={t.mean():.4f}  control={c.mean():.4f}")
        print(f"    matched ratio={ratio:.3f}  p_perm={p:.4f}")
        print(f"    naive: {naive_r:.2f} p={naive_p:.3f}  ->  {verdict}\n")
        results_rows.append({"metric": metric, "n_matched": len(t), "treated_mean": t.mean(), "control_mean": c.mean(),
                             "matched_ratio": ratio, "p_perm": p, "verdict": verdict})
    RESULTS_DIR = ROOT / "data" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_rows).to_csv(RESULTS_DIR / "nextgen_propensity_results.csv", index=False)


if __name__ == "__main__":
    main()
