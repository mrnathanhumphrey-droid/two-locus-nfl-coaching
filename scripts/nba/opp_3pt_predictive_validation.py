"""NBA opp_3PT_rate predictive validation — does incoming HC's prior-tenure
opp_3PT_rate profile forecast the new team's opp_3PT_rate shift?

Mirrors NFL OC predictive validation methodology but on the NBA defensive
perimeter scheme metric that just survived propensity matching.

If r > 0.20 with p < 0.05, NBA defensive perimeter scheme is portable across
team contexts — coaches bring opp 3PT-rate tendency with them.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "nba" / "inputs"   # staged per mode: scripts/build/stage_nba.py
RESULTS = ROOT / "data" / "results"

K = 10
N_PERM = 5000
SEED = 42
METRIC = "opp_three_pt_rate_residual"


def build_residuals():
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    tg = (bs.groupby(["game_id", "season", "team_abbr", "game_date"],
                       as_index=False)
            .agg(FGM=("FGM", "sum"), FGA=("FGA", "sum"), FG3A=("FG3A", "sum")))
    tg["three_pt_rate"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    opp = tg.rename(columns={"team_abbr": "opp_team",
                              "three_pt_rate": "opp_three_pt_rate"})[
        ["game_id", "opp_team", "opp_three_pt_rate"]]
    merged = tg.merge(opp, on="game_id", how="left")
    merged = merged[merged["opp_team"] != merged["team_abbr"]].copy()
    smean = merged.groupby("season")["opp_three_pt_rate"].transform("mean")
    merged["opp_three_pt_rate_residual"] = merged["opp_three_pt_rate"] - smean
    merged = merged.sort_values(["season", "team_abbr", "game_date"])
    merged["game_in_season"] = merged.groupby(["season", "team_abbr"]).cumcount() + 1
    return merged


def load_staff():
    flags = pd.read_parquet(NBA_PROJ / "coaching_change_flags.parquet")
    return flags[["team_abbr", "season", "start_of_season_name"]].rename(
        columns={"start_of_season_name": "HC"})


def find_retread(staff):
    s = staff[staff["HC"].notna()].copy().sort_values(["HC", "season"]).reset_index(drop=True)
    events = []
    for hc, g in s.groupby("HC"):
        g = g.sort_values("season").reset_index(drop=True)
        if len(g) < 2: continue
        for i in range(1, len(g)):
            if g.loc[i, "team_abbr"] != g.loc[i-1, "team_abbr"]:
                events.append({
                    "HC": hc,
                    "prior_team": g.loc[i-1, "team_abbr"],
                    "prior_season": g.loc[i-1, "season"],
                    "new_team": g.loc[i, "team_abbr"],
                    "new_season": g.loc[i, "season"],
                })
    return pd.DataFrame(events)


def compute_profile(resid, team, season, k, anchor):
    g = resid[(resid["team_abbr"] == team) & (resid["season"] == season)].sort_values("game_in_season")
    if len(g) < k: return np.nan
    sub = g.iloc[-k:] if anchor == "last" else g.iloc[:k]
    return float(sub[METRIC].mean())


def perm_test_corr(x, y, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = float(np.corrcoef(x, y)[0, 1])
    null = np.zeros(n)
    for i in range(n):
        null[i] = float(np.corrcoef(x, rng.permutation(y))[0, 1])
    return obs, float((null >= obs).mean()), float((np.abs(null) >= abs(obs)).mean())


def fit_ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ b
    ss_r = float(((y - pred) ** 2).sum())
    ss_t = float(((y - y.mean()) ** 2).sum())
    return float(b[0]), float(b[1]), (1 - ss_r / ss_t if ss_t > 0 else 0.0)


def main():
    print(f"=== NBA OPP_3PT_RATE PREDICTIVE VALIDATION ===")
    print(f"  metric={METRIC}, K={K}\n")
    staff = load_staff()
    resid = build_residuals()
    events = find_retread(staff)
    print(f"retread HC events: {len(events)}")

    rows = []
    for _, ev in events.iterrows():
        p_prior = compute_profile(resid, ev["prior_team"], ev["prior_season"], K, "last")
        # New team's "pre" is new_team in (new_season - 1)
        ny = int(ev["new_season"].split("-")[0])
        new_pre_season = f"{ny - 1}-{str(ny % 100).zfill(2)}"
        p_pre = compute_profile(resid, ev["new_team"], new_pre_season, K, "last")
        p_post = compute_profile(resid, ev["new_team"], ev["new_season"], K, "first")
        if any(np.isnan([p_prior, p_pre, p_post])): continue
        rows.append({
            "HC": ev["HC"],
            "prior_team": ev["prior_team"], "prior_season": ev["prior_season"],
            "new_team": ev["new_team"], "new_season": ev["new_season"],
            "shift_obs": p_post - p_pre,
            "shift_pred": p_prior - p_pre,
        })
    paired = pd.DataFrame(rows)
    print(f"events with all 3 profiles computable: {len(paired)}")
    if len(paired) < 5:
        print("INSUFFICIENT"); return

    x = paired["shift_pred"].values
    y = paired["shift_obs"].values
    r, p1, p2 = perm_test_corr(x, y)
    intercept, slope, r2 = fit_ols(x, y)
    agree = int(((paired["shift_pred"] > 0) == (paired["shift_obs"] > 0)).sum())

    print(f"\n=== HEADLINE ===")
    print(f"  n events:                {len(paired)}")
    print(f"  Pearson r:               {r:+.4f}")
    print(f"  p_one_sided:             {p1:.4f}")
    print(f"  p_two_sided:             {p2:.4f}")
    print(f"  OLS slope:               {slope:+.4f}")
    print(f"  OLS R^2:                 {r2:.4f}")
    print(f"  direction agreement:     {agree}/{len(paired)} ({100*agree/len(paired):.0f}%)")

    if p1 < 0.05 and r > 0.2:
        verdict = "PORTABLE - defensive perimeter scheme forecasts shift"
    elif p1 < 0.05:
        verdict = "WEAK PORTABLE"
    elif r > 0.2:
        verdict = "MARGINAL"
    else:
        verdict = "NOT PORTABLE"
    print(f"  VERDICT: {verdict}")

    print(f"\n  Cross-reference NFL OC RZ-shotgun (offensive perimeter analog):")
    print(f"    n=149, r=+0.48, p<0.0001, slope=0.45")

    paired.to_csv(RESULTS / "opp_3pt_predictive_validation_paired.csv", index=False)
    pd.DataFrame([{"test": "NBA opp_3pt_rate portability",
                   "n_events": len(paired), "pearson_r": r,
                   "p_one_sided": p1, "p_two_sided": p2,
                   "ols_slope": slope, "ols_r2": r2,
                   "direction_agreement": agree / len(paired),
                   "verdict": verdict}]).to_csv(
        RESULTS / "opp_3pt_predictive_validation_summary.csv", index=False)
    print(f"\n  wrote results CSVs")

    if len(paired) >= 5:
        paired["abs_obs"] = paired["shift_obs"].abs()
        print(f"\n  Top 5 by |observed shift|:")
        print(paired.nlargest(5, "abs_obs")[
            ["HC", "prior_team", "prior_season", "new_team", "new_season",
              "shift_pred", "shift_obs"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
