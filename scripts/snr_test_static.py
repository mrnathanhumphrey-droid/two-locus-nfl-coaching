"""SNR test — STATIC LAYER.

Pre-registered hypothesis (from the brief):
  "Coaching staff is the load-bearing latent — not team-id, not individual player."
  Pooling axis = coaching staff vector. Same Collatz-protocol SNR diagnostic
  that locked an analogous basketball projection system.

This test, the static layer:
  - For each team-season, aggregate per-game state_residuals
    (pass_rate_residual, plays_residual, red_zone_trips_residual) to mean
    + EPA/play residual
  - Partial out nuisance covariates: roster_continuity_pct, injury totals
  - Compare partition SNR for two candidate structural classes:
      A. coaching_vector_id   (hash of (HC, OC, DC) tuple per team-season)
      B. team_id              (the obvious baseline)
  - For each residual dimension and each class, compute:
      SS_between = sum over groups of n_g * (group_mean - grand_mean)^2
      SS_within  = sum over rows of (x - group_mean)^2
      SNR        = SS_between / SS_within
      eta_squared = SS_between / SS_total  (variance explained, comparable across classes)

Pre-registered thresholds:
  SNR >= 1.5 -> class carries signal
  SNR(coaching) > SNR(team_id) by >=1.5x -> coaching is the load-bearing latent
  SNR(coaching) < SNR(team_id) -> coaching does not transfer; walk-back triggered

Walk-back condition: if static layer rejects, the central hypothesis fails
on the cleanest test. Within-season + within-game tests would still be
informative but the headline result is rejection.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data"


def partial_out(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y after OLS on X (with intercept)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def snr_partition(values: np.ndarray, groups: np.ndarray) -> dict:
    """Compute SNR for partition of `values` by `groups`."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if len(df) == 0:
        return {"snr": np.nan, "eta_sq": np.nan, "n_groups": 0,
                "ss_between": np.nan, "ss_within": np.nan, "ss_total": np.nan}
    grand = df["v"].mean()
    grp_means = df.groupby("g")["v"].mean()
    grp_sizes = df.groupby("g")["v"].size()
    ss_between = float(((grp_means - grand) ** 2 * grp_sizes).sum())
    ss_total = float(((df["v"] - grand) ** 2).sum())
    ss_within = ss_total - ss_between
    snr = ss_between / ss_within if ss_within > 0 else np.nan
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    return {"snr": snr, "eta_sq": eta_sq, "n_groups": int(df["g"].nunique()),
            "ss_between": ss_between, "ss_within": ss_within, "ss_total": ss_total,
            "n": len(df)}


def main():
    print("=== loading data ===")
    sr = pd.read_parquet(ROOT / "games" / "state_residuals.parquet")
    games = pd.read_parquet(ROOT / "games" / "games.parquet")
    cs = pd.read_parquet(ROOT / "coaching" / "coach_seasons.parquet")
    print(f"  state_residuals: {len(sr):,} team-game rows ({sr['season'].min()}-{sr['season'].max()})")

    # 1) Aggregate per-game residuals to (team, season)
    ts = sr.groupby(["posteam", "season"]).agg(
        pass_rate_resid=("pass_rate_residual", "mean"),
        plays_resid=("plays_residual", "mean"),
        rz_trips_resid=("red_zone_trips_residual", "mean"),
        epa_per_play=("epa_per_play", "mean"),
        n_games=("game_id", "count"),
    ).reset_index().rename(columns={"posteam": "team"})
    print(f"  team-seasons: {len(ts):,}")

    # 2) Build coaching_vector_id from (HC, OC, DC) per team-season
    #    For each team-season, get the primary HC/OC/DC name (handle midseason
    #    changes by taking the coach with most games, or first if tie)
    cs["games_filled"] = cs["games"].fillna(0)
    primary = (cs[cs["role"].isin(["head_coach", "offensive_coordinator", "defensive_coordinator"])]
               .sort_values(["team", "season", "role", "games_filled"], ascending=[True, True, True, False])
               .groupby(["team", "season", "role"]).first().reset_index())
    staff = primary.pivot_table(index=["team", "season"], columns="role",
                                 values="coach_name", aggfunc="first").reset_index()
    staff["coaching_vector_id"] = (staff.get("head_coach", "").fillna("") + "|"
                                    + staff.get("offensive_coordinator", "").fillna("") + "|"
                                    + staff.get("defensive_coordinator", "").fillna(""))

    # Need to translate coaching CS team abbr (PFR) to nfl game team abbr
    # PFR uses: rai/rav/oti/sdg/htx/clt/crd/ram (vs NFL: LV/BAL/TEN/LAC/HOU/IND/ARI/LA)
    # For state_residuals the team col is from pbp (NFL standard abbrs); we need to
    # map staff PFR abbr -> NFL abbr for the join.
    pfr_to_nfl = {
        "rai": "LV", "rav": "BAL", "oti": "TEN", "sdg": "LAC", "htx": "HOU",
        "clt": "IND", "crd": "ARI", "ram": "LA", "buf": "BUF", "mia": "MIA",
        "nwe": "NE", "nyj": "NYJ", "cin": "CIN", "cle": "CLE", "pit": "PIT",
        "jax": "JAX", "den": "DEN", "kan": "KC", "dal": "DAL", "nyg": "NYG",
        "phi": "PHI", "was": "WAS", "chi": "CHI", "det": "DET", "gnb": "GB",
        "min": "MIN", "atl": "ATL", "car": "CAR", "nor": "NO", "tam": "TB",
        "sea": "SEA", "sfo": "SF",
    }
    # Some legacy abbreviations also appear pre-relocation. Add variants:
    legacy_remap = {
        "STL": "LA",    # St. Louis Rams -> LA Rams (current franchise)
        "SD":  "LAC",   # San Diego Chargers -> LA Chargers
        "OAK": "LV",    # Oakland Raiders -> Las Vegas Raiders
        "LAR": "LA",    # alt LA Rams
    }
    staff["team_nfl"] = staff["team"].map(pfr_to_nfl)
    staff = staff.rename(columns={"team": "team_pfr"})
    # Build a mapping that handles legacy abbrs in state_residuals
    ts["team_canon"] = ts["team"].map(legacy_remap).fillna(ts["team"])

    merged = ts.merge(staff[["team_nfl", "season", "head_coach",
                              "offensive_coordinator", "defensive_coordinator",
                              "coaching_vector_id"]],
                       left_on=["team_canon", "season"], right_on=["team_nfl", "season"],
                       how="left")
    matched = merged["coaching_vector_id"].notna().sum()
    print(f"  matched team-seasons to coaching staff: {matched}/{len(merged)} "
          f"({100*matched/len(merged):.1f}%)")

    # 3) Add nuisance covariates: roster_continuity, injuries
    #    Continuity table is keyed on team (NFL std abbr) + season
    cont = pd.read_parquet(ROOT / "rosters" / "continuity.parquet")
    merged = merged.merge(cont[["team", "season", "roster_continuity_pct"]],
                           left_on=["team_canon", "season"], right_on=["team", "season"],
                           how="left")
    print(f"  continuity attached: {merged['roster_continuity_pct'].notna().sum()}/{len(merged)}")

    # team-season aggregate of injury Q/Out from games (already per-week, aggregate)
    # Use home_inj_out from games joined to home_team-season ; since games has both
    # home and away rows aggregate by team
    inj_h = games.dropna(subset=["home_inj_out"])[["season", "home_team", "home_inj_out", "home_inj_q"]].rename(
        columns={"home_team": "team", "home_inj_out": "out", "home_inj_q": "q"})
    inj_a = games.dropna(subset=["away_inj_out"])[["season", "away_team", "away_inj_out", "away_inj_q"]].rename(
        columns={"away_team": "team", "away_inj_out": "out", "away_inj_q": "q"})
    inj_team_season = (pd.concat([inj_h, inj_a]).groupby(["team", "season"])
                          .agg(inj_out_total=("out", "sum"), inj_q_total=("q", "sum"))
                          .reset_index())
    merged = merged.merge(inj_team_season, left_on=["team_canon", "season"],
                           right_on=["team", "season"], how="left", suffixes=("", "_inj"))
    print(f"  injury totals attached: {merged['inj_out_total'].notna().sum()}/{len(merged)}")

    # 4) Build the test set: rows where we have coaching_vector + continuity + injury
    test = merged.dropna(subset=["coaching_vector_id", "roster_continuity_pct"]).copy()
    # injury covariates only exist 2009+; fill missing with 0 + flag
    test["inj_out_total"] = test["inj_out_total"].fillna(0)
    test["inj_q_total"] = test["inj_q_total"].fillna(0)
    test["has_inj"] = (test["season"] >= 2009).astype(float)
    print(f"\n  test set: {len(test):,} team-seasons")
    print(f"  unique teams: {test['team_canon'].nunique()}")
    print(f"  unique coaching vectors: {test['coaching_vector_id'].nunique()}")
    print(f"  seasons: {test['season'].min()}-{test['season'].max()}")

    # 5) Partial out nuisance covariates from each residual dimension
    nuisance_cols = ["roster_continuity_pct", "inj_out_total", "inj_q_total", "has_inj"]
    X_nuisance = test[nuisance_cols].values
    residual_cols = ["pass_rate_resid", "plays_resid", "rz_trips_resid", "epa_per_play"]

    print("\n=== partialled residuals (after nuisance partial-out) ===")
    partialled = {}
    for col in residual_cols:
        y = test[col].values.astype(float)
        valid = ~np.isnan(y)
        y_clean = y[valid]
        X_clean = X_nuisance[valid]
        resid = partial_out(y_clean, X_clean)
        out_arr = np.full(len(y), np.nan)
        out_arr[valid] = resid
        partialled[col] = out_arr
        print(f"  {col}: var pre-partial={np.nanvar(y):.5f}  post-partial={np.nanvar(resid):.5f}")

    # 6) SNR for each (residual_dim, class) combo
    print("\n=== SNR partition test ===")
    print(f"  classes:")
    print(f"    A. coaching_vector  ({test['coaching_vector_id'].nunique()} groups)")
    print(f"    B. team_id          ({test['team_canon'].nunique()} groups)")
    print()
    print(f"  {'residual_dim':<22} {'class':<22} {'n':>4} {'n_grp':>6} "
          f"{'snr':>10} {'eta_sq':>10} {'verdict':>10}")
    print(f"  {'-'*22} {'-'*22} {'-'*4} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    rows = []
    for col in residual_cols:
        y = partialled[col]
        for cls_name, cls_col in [("coaching_vector", "coaching_vector_id"),
                                   ("team_id", "team_canon")]:
            r = snr_partition(y, test[cls_col].values)
            verdict = "DISCRETE" if r["snr"] >= 1.5 else "GRADED" if r["snr"] >= 0.3 else "NULL"
            rows.append({"residual_dim": col, "class": cls_name, **r, "verdict": verdict})
            print(f"  {col:<22} {cls_name:<22} {r.get('n', 0):>4} "
                  f"{r['n_groups']:>6} {r['snr']:>10.4f} {r['eta_sq']:>10.4f} "
                  f"{verdict:>10}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(ROOT / "snr_test_static_results.csv", index=False)

    # 7) Head-to-head comparison
    print("\n=== HEAD-TO-HEAD: coaching_vector vs team_id ===")
    print(f"  {'residual_dim':<22} {'snr_coaching':>14} {'snr_team':>14} "
          f"{'ratio':>8} {'verdict':>20}")
    summary_rows = []
    for col in residual_cols:
        c = next(r for r in rows if r["residual_dim"] == col and r["class"] == "coaching_vector")
        t = next(r for r in rows if r["residual_dim"] == col and r["class"] == "team_id")
        ratio = c["snr"] / t["snr"] if t["snr"] > 0 else np.nan
        if c["snr"] >= 1.5 and ratio >= 1.5:
            verdict = "COACHING WINS"
        elif c["snr"] >= 0.3 and ratio >= 1.5:
            verdict = "COACHING LEADS (graded)"
        elif t["snr"] > c["snr"]:
            verdict = "TEAM-ID WINS"
        elif c["snr"] < 0.3:
            verdict = "BOTH NULL"
        else:
            verdict = "TIE"
        summary_rows.append({"residual_dim": col, "snr_coaching": c["snr"],
                              "snr_team": t["snr"], "ratio": ratio, "verdict": verdict})
        print(f"  {col:<22} {c['snr']:>14.4f} {t['snr']:>14.4f} "
              f"{ratio:>8.3f} {verdict:>20}")
    pd.DataFrame(summary_rows).to_csv(ROOT / "snr_test_static_h2h.csv", index=False)

    print("\n=== VERDICT ===")
    coaching_wins = sum(1 for r in summary_rows if "COACHING" in r["verdict"])
    team_wins = sum(1 for r in summary_rows if "TEAM" in r["verdict"])
    nulls = sum(1 for r in summary_rows if "NULL" in r["verdict"])
    print(f"  coaching wins/leads: {coaching_wins}/{len(summary_rows)}")
    print(f"  team-id wins:        {team_wins}/{len(summary_rows)}")
    print(f"  both null:           {nulls}/{len(summary_rows)}")
    if coaching_wins >= 3:
        print("  >>> CENTRAL HYPOTHESIS SUPPORTED on majority of residual dimensions")
    elif team_wins >= 3:
        print("  >>> WALK-BACK: coaching does not partition residuals better than team-id")
    elif nulls >= 3:
        print("  >>> NULL: neither class carries signal at the static layer")
    else:
        print("  >>> MIXED: results vary by residual dimension")


if __name__ == "__main__":
    main()
