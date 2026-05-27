"""SNR test — SVI-analogue (decomposing team-id into coaching-history + non-coaching).

Question: is team_id confounded with averaged-coaching-over-time?

Test design:
  1. Build per-team-season coaching-history features from coach_seasons.parquet:
       hc_tenure_years            consecutive seasons current HC
       hc_count_5yr_window        distinct HCs in prior 5 years
       oc_continuity_with_hc      OC same as last yr given HC same
       dc_continuity_with_hc      DC same as last yr given HC same
       cumulative_unique_hcs      cumulative HC churn since 1999
       hc_first_year              binary: HC's first year w/ this team
       coordinator_change_count   distinct OC+DC pairs over team's history
  2. For each residual dim (pass_rate / plays / rz_trips / epa_per_play
     post-partialling-out continuity + injury):
       a. Block A: coaching-history features ONLY (regression R² = R2_A)
       b. Block B: team_id one-hot ONLY (regression R² = R2_B)
       c. Block A+B: combined (R² = R2_AB)
       d. Compute incremental:
            R²_A_given_B = R2_AB - R2_B   (info coaching adds beyond team-id)
            R²_B_given_A = R2_AB - R2_A   (info team-id adds beyond coaching-history)
  3. Direct test: regress team_id-only residual on coaching-history features.
     R² of that = "fraction of team-id's signal that's actually
     coaching-history-explained."

If team-id is mostly coaching averaged over time:
  R²_B_given_A is small  AND  team-id residual is well-predicted by coaching-history
  -> SVI confound confirmed
If team-id has substantial non-coaching content:
  R²_B_given_A is meaningful AND coaching-history poorly predicts team-id residual
  -> SVI confound rejected; team-id has independent organizational variance
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"


def partial_out(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def r2(y, X):
    """OLS R² with intercept."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return 0.0
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def build_coaching_history():
    """Per (team_pfr, season): coaching history features."""
    cs = pd.read_parquet(ROOT / "coaching" / "coach_seasons.parquet")
    cs["games_filled"] = cs["games"].fillna(0)
    primary = (cs[cs["role"].isin(["head_coach", "offensive_coordinator", "defensive_coordinator"])]
               .sort_values(["team", "season", "role", "games_filled"], ascending=[True, True, True, False])
               .groupby(["team", "season", "role"]).first().reset_index())
    staff = primary.pivot_table(index=["team", "season"], columns="role",
                                 values="coach_name", aggfunc="first").reset_index()
    staff = staff.rename(columns={"head_coach": "hc", "offensive_coordinator": "oc",
                                    "defensive_coordinator": "dc"})
    rows = []
    for tm, grp in staff.groupby("team"):
        grp = grp.sort_values("season").reset_index(drop=True)
        unique_hcs_so_far = set()
        unique_oc_dc_pairs = set()
        for i in range(len(grp)):
            r = grp.iloc[i]
            hc = r.get("hc", "")
            oc = r.get("oc", "")
            dc = r.get("dc", "")
            # tenure: count back to find first year of current HC
            tenure = 1
            for j in range(i - 1, -1, -1):
                if grp.iloc[j].get("hc", "") == hc:
                    tenure += 1
                else:
                    break
            # 5-year HC turnover (incl. current)
            window = grp.iloc[max(0, i - 4): i + 1]
            hc_5yr = window["hc"].nunique() if "hc" in window.columns else 1
            # continuity flags
            prev = grp.iloc[i - 1] if i > 0 else None
            hc_first_year = 1 if (prev is None or prev.get("hc", "") != hc) else 0
            oc_cont = 1 if (prev is not None
                            and prev.get("hc", "") == hc
                            and prev.get("oc", "") == oc
                            and oc != "") else 0
            dc_cont = 1 if (prev is not None
                            and prev.get("hc", "") == hc
                            and prev.get("dc", "") == dc
                            and dc != "") else 0
            unique_hcs_so_far.add(hc)
            unique_oc_dc_pairs.add((oc, dc))
            rows.append({
                "team_pfr": tm, "season": int(r["season"]),
                "hc_tenure_years": tenure,
                "hc_count_5yr_window": int(hc_5yr),
                "hc_first_year": hc_first_year,
                "oc_continuity_with_hc": oc_cont,
                "dc_continuity_with_hc": dc_cont,
                "cumulative_unique_hcs": len(unique_hcs_so_far),
                "cumulative_unique_oc_dc_pairs": len(unique_oc_dc_pairs),
            })
    return pd.DataFrame(rows)


def build_test_set():
    """Reproduce the static-layer partialled residuals + add coaching-history."""
    sr = pd.read_parquet(ROOT / "games" / "state_residuals.parquet")
    games = pd.read_parquet(ROOT / "games" / "games.parquet")
    cont = pd.read_parquet(ROOT / "rosters" / "continuity.parquet")

    ts = sr.groupby(["posteam", "season"]).agg(
        pass_rate_resid=("pass_rate_residual", "mean"),
        plays_resid=("plays_residual", "mean"),
        rz_trips_resid=("red_zone_trips_residual", "mean"),
        epa_per_play=("epa_per_play", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    legacy_remap = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA"}
    ts["team_canon"] = ts["team"].map(legacy_remap).fillna(ts["team"])

    # PFR -> NFL mapping
    pfr_to_nfl = {
        "rai": "LV", "rav": "BAL", "oti": "TEN", "sdg": "LAC", "htx": "HOU",
        "clt": "IND", "crd": "ARI", "ram": "LA", "buf": "BUF", "mia": "MIA",
        "nwe": "NE", "nyj": "NYJ", "cin": "CIN", "cle": "CLE", "pit": "PIT",
        "jax": "JAX", "den": "DEN", "kan": "KC", "dal": "DAL", "nyg": "NYG",
        "phi": "PHI", "was": "WAS", "chi": "CHI", "det": "DET", "gnb": "GB",
        "min": "MIN", "atl": "ATL", "car": "CAR", "nor": "NO", "tam": "TB",
        "sea": "SEA", "sfo": "SF",
    }
    ch = build_coaching_history()
    ch["team_nfl"] = ch["team_pfr"].map(pfr_to_nfl)
    ts = ts.merge(ch, left_on=["team_canon", "season"],
                   right_on=["team_nfl", "season"], how="left")
    ts = ts.merge(cont[["team", "season", "roster_continuity_pct"]],
                   left_on=["team_canon", "season"],
                   right_on=["team", "season"], how="left", suffixes=("", "_c"))

    inj_h = games.dropna(subset=["home_inj_out"])[["season", "home_team", "home_inj_out", "home_inj_q"]].rename(
        columns={"home_team": "team", "home_inj_out": "out", "home_inj_q": "q"})
    inj_a = games.dropna(subset=["away_inj_out"])[["season", "away_team", "away_inj_out", "away_inj_q"]].rename(
        columns={"away_team": "team", "away_inj_out": "out", "away_inj_q": "q"})
    inj_team_season = (pd.concat([inj_h, inj_a]).groupby(["team", "season"])
                          .agg(inj_out_total=("out", "sum"), inj_q_total=("q", "sum")).reset_index())
    ts = ts.merge(inj_team_season, left_on=["team_canon", "season"],
                   right_on=["team", "season"], how="left", suffixes=("", "_inj"))

    keep = ts.dropna(subset=["hc_tenure_years", "roster_continuity_pct"]).copy()
    keep["inj_out_total"] = keep["inj_out_total"].fillna(0)
    keep["inj_q_total"] = keep["inj_q_total"].fillna(0)
    keep["has_inj"] = (keep["season"] >= 2009).astype(float)

    # partial out nuisance from each residual dim
    nuisance = keep[["roster_continuity_pct", "inj_out_total", "inj_q_total", "has_inj"]].values
    for col in ["pass_rate_resid", "plays_resid", "rz_trips_resid", "epa_per_play"]:
        y = keep[col].values.astype(float)
        valid = ~np.isnan(y)
        out = np.full(len(y), np.nan)
        out[valid] = partial_out(y[valid], nuisance[valid])
        keep[col + "_partialled"] = out
    return keep


def main():
    print("=== building test set + coaching-history features ===")
    df = build_test_set()
    print(f"  test set: {len(df)} team-seasons "
          f"({df['season'].min()}-{df['season'].max()})")

    coach_hist_cols = ["hc_tenure_years", "hc_count_5yr_window", "hc_first_year",
                       "oc_continuity_with_hc", "dc_continuity_with_hc",
                       "cumulative_unique_hcs", "cumulative_unique_oc_dc_pairs"]
    team_dummies = pd.get_dummies(df["team_canon"], drop_first=True).astype(float).values

    print(f"\n  coaching-history features: {len(coach_hist_cols)}")
    print(f"  team-id dummies: {team_dummies.shape[1]}")

    print("\n=== variance decomposition per residual dim ===")
    print(f"  {'dim':<22} {'R2_A_coach':>12} {'R2_B_team':>12} {'R2_AB_both':>12} "
          f"{'A_given_B':>12} {'B_given_A':>12} {'overlap':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    rows = []
    for col in ["pass_rate_resid", "plays_resid", "rz_trips_resid", "epa_per_play"]:
        y = df[col + "_partialled"].values
        valid = ~np.isnan(y)
        y_v = y[valid]
        X_coach = df[coach_hist_cols].values[valid].astype(float)
        X_team = team_dummies[valid]
        X_both = np.column_stack([X_coach, X_team])
        r2_A = r2(y_v, X_coach)
        r2_B = r2(y_v, X_team)
        r2_AB = r2(y_v, X_both)
        a_given_b = r2_AB - r2_B
        b_given_a = r2_AB - r2_A
        overlap = r2_A + r2_B - r2_AB
        rows.append({"dim": col, "r2_coach": r2_A, "r2_team": r2_B,
                     "r2_both": r2_AB, "a_given_b": a_given_b,
                     "b_given_a": b_given_a, "overlap": overlap})
        print(f"  {col:<22} {r2_A:>12.4f} {r2_B:>12.4f} {r2_AB:>12.4f} "
              f"{a_given_b:>12.4f} {b_given_a:>12.4f} {overlap:>10.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "snr_svi_analogue_decomposition.csv", index=False)

    # Direct SVI test: regress team-id-only fitted values onto coaching history
    print("\n=== DIRECT SVI TEST: how much of team-id's signal is coaching-history? ===")
    print(f"  {'dim':<22} {'team_id_R2':>12} {'team_resid_explained_by_coach':>30}")
    print(f"  {'-'*22} {'-'*12} {'-'*30}")
    direct_rows = []
    for col in ["pass_rate_resid", "plays_resid", "rz_trips_resid", "epa_per_play"]:
        y = df[col + "_partialled"].values
        valid = ~np.isnan(y)
        y_v = y[valid]
        X_team = team_dummies[valid]
        # fit team-id only -> get the team-id explained component
        X1 = np.column_stack([np.ones(len(X_team)), X_team])
        beta, *_ = np.linalg.lstsq(X1, y_v, rcond=None)
        team_id_fitted = X1 @ beta  # the team-id signal
        # Now explain team_id_fitted using coaching-history features
        X_coach = df[coach_hist_cols].values[valid].astype(float)
        # need to also include team's mean coaching history per team -- but
        # coaching-history features are already per (team, year) so this works
        explained = r2(team_id_fitted, X_coach)
        team_only_r2 = r2(y_v, X_team)
        direct_rows.append({"dim": col, "team_id_R2": team_only_r2,
                            "team_id_signal_R2_by_coach_history": explained})
        print(f"  {col:<22} {team_only_r2:>12.4f} {explained:>30.4f}")
    pd.DataFrame(direct_rows).to_csv(ROOT / "snr_svi_direct_test.csv", index=False)

    # Verdict
    print("\n=== VERDICT ===")
    print("  R²(coaching|team) — info coaching adds beyond team-id")
    print("  R²(team|coaching) — info team-id adds beyond coaching-history")
    print()
    for r in rows:
        if r["b_given_a"] > 0.10 and r["a_given_b"] < 0.05:
            v = "TEAM-ID has independent non-coaching signal (SVI confound DOES NOT FULLY EXPLAIN)"
        elif r["b_given_a"] < 0.05 and r["a_given_b"] < 0.05:
            v = "Both already absorbed into nuisance; no marginal info"
        elif r["a_given_b"] > 0.10 and r["b_given_a"] < 0.05:
            v = "COACHING-HISTORY has signal team-id misses (rare; usually means dynamic features matter)"
        elif r["overlap"] > max(r["a_given_b"], r["b_given_a"]):
            v = "OVERLAP DOMINATES: team-id and coaching-history are largely the same signal"
        else:
            v = "MIXED / partial separability"
        print(f"  {r['dim']:<22} -> {v}")


if __name__ == "__main__":
    main()
