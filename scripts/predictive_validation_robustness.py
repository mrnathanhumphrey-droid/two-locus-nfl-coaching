"""Robustness checks for HC + OC predictive validation slopes.

Block (i) — Placebo: for continuous-staff team-seasons (no HC/OC change),
substitute a randomly drawn coach prior from the same season's coaching
pool. Re-compute the predicted-vs-observed slope across B=1000 draws.
Report median + 95% interval + percentile of observed slope.

Block (ii) — Influence: drop Reid PHI 2012 → KC 2013 from HC retread set
and refit. Verify HC slope + R² remain positive.

Constructs mirror the original validations:
  HC uses pass_rate_residual, K=8 games
  OC uses rz_shotgun_rate_residual, K=8 games
  shift_pred = profile_coach_prior - profile_team_pre
  shift_obs  = profile_team_post - profile_team_pre

Continuous-staff placebo:
  Same coach in season S-1 and season S. shift_obs computed within same
  team using last-8-of-S-1 -> first-8-of-S. Placebo shift_pred substitutes
  a random retread coach's prior (sampled from the same season's pool of
  retread events).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_change_event_study import LEGACY_REMAP, build_staff_panel  # noqa: E402
from oc_rz_focused_followup import build_rz_residuals  # noqa: E402
from hc_predictive_validation import compute_profile  # noqa: E402
from oc_predictive_validation import compute_oc_profile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data"
K = 8
B = 1000
SEED = 2026


def fit_ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(beta[0]), float(beta[1]), float(r2)


def perm_p(x, y, observed_r, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        ys = rng.permutation(y)
        null[i] = float(np.corrcoef(x, ys)[0, 1])
    return float((null >= observed_r).mean())


def load_offensive_residuals():
    sr = pd.read_parquet(ROOT / "games" / "state_residuals.parquet")
    if "week" not in sr.columns:
        games = pd.read_parquet(ROOT / "games" / "games.parquet")
        sr = sr.merge(games[["game_id", "week"]], on="game_id", how="left")
    sr["team_canon"] = sr["posteam"].map(LEGACY_REMAP).fillna(sr["posteam"])
    return sr


def continuous_staff_seasons(staff, role_col):
    """Identify (team, season) where the same person held role_col in season-1 and season."""
    s = staff.sort_values(["team_nfl", "season"]).reset_index(drop=True)
    rows = []
    for tm, g in s.groupby("team_nfl"):
        g = g.sort_values("season").reset_index(drop=True)
        for i in range(1, len(g)):
            prev = g.loc[i - 1]
            cur = g.loc[i]
            if pd.isna(prev[role_col]) or pd.isna(cur[role_col]):
                continue
            if prev[role_col] == cur[role_col] and cur["season"] == prev["season"] + 1:
                rows.append({
                    "team_nfl": tm,
                    "person": cur[role_col],
                    "prior_season": int(prev["season"]),
                    "new_season": int(cur["season"]),
                })
    return pd.DataFrame(rows)


def main():
    print(f"=== Predictive Validation Robustness (B={B} placebo draws, seed={SEED}) ===\n")
    rng = np.random.default_rng(SEED)
    staff = build_staff_panel()
    print(f"staff panel rows: {len(staff)}")

    # ============================================================
    # HC PLACEBO
    # ============================================================
    print("\n" + "=" * 70)
    print("BLOCK (i) HC PLACEBO")
    print("=" * 70)
    hc_obs = pd.read_csv(ROOT / "hc_predictive_validation_paired.csv")
    hc_obs_intercept, hc_obs_slope, hc_obs_r2 = fit_ols(
        hc_obs["shift_pred"].values, hc_obs["shift_obs"].values)
    print(f"  HC observed slope: {hc_obs_slope:+.4f}  R²: {hc_obs_r2:.4f}  n: {len(hc_obs)}")

    sr = load_offensive_residuals()
    cont_hc = continuous_staff_seasons(staff, "HC")
    print(f"  continuous-staff HC team-seasons candidates: {len(cont_hc)}")

    # Compute shift_obs per continuous-staff team-season
    cont_rows = []
    for _, r in cont_hc.iterrows():
        pre = compute_profile(sr, r["team_nfl"], r["prior_season"], K, anchor="last")
        post = compute_profile(sr, r["team_nfl"], r["new_season"], K, anchor="first")
        if not np.isnan(pre) and not np.isnan(post):
            cont_rows.append({
                "team_nfl": r["team_nfl"], "new_season": r["new_season"],
                "profile_pre": pre, "profile_post": post,
                "shift_obs_placebo": post - pre,
            })
    cont_df = pd.DataFrame(cont_rows)
    print(f"  continuous-staff team-seasons with computable shift: {len(cont_df)}")

    # Pool of coach priors per season (from retread events)
    # Pool = profile_hc_prior values, indexed by NEW_SEASON of the retread event
    # (i.e., the season the prior coach was hired into — used as 'same season's coaching pool')
    hc_pool_by_season = {}
    for s in sorted(hc_obs["new_season"].unique()):
        hc_pool_by_season[s] = hc_obs[hc_obs["new_season"] == s]["profile_hc_prior"].values
    # Broader fallback pool across all seasons
    broad_hc_pool = hc_obs["profile_hc_prior"].values
    print(f"  HC same-season pool sizes (min/median/max): "
          f"{min(len(v) for v in hc_pool_by_season.values())}/"
          f"{int(np.median([len(v) for v in hc_pool_by_season.values()]))}/"
          f"{max(len(v) for v in hc_pool_by_season.values())}")

    placebo_slopes_hc = np.empty(B)
    for b in range(B):
        sampled_priors = []
        for _, row in cont_df.iterrows():
            pool = hc_pool_by_season.get(int(row["new_season"]), broad_hc_pool)
            if len(pool) == 0:
                pool = broad_hc_pool
            sampled_priors.append(rng.choice(pool))
        sampled_priors = np.array(sampled_priors)
        shift_pred_placebo = sampled_priors - cont_df["profile_pre"].values
        shift_obs_arr = cont_df["shift_obs_placebo"].values
        _, slope_b, _ = fit_ols(shift_pred_placebo, shift_obs_arr)
        placebo_slopes_hc[b] = slope_b
    med_hc = float(np.median(placebo_slopes_hc))
    lo_hc, hi_hc = np.percentile(placebo_slopes_hc, [2.5, 97.5])
    pct_hc = float((placebo_slopes_hc < hc_obs_slope).mean() * 100.0)
    print(f"\n  HC PLACEBO RESULTS")
    print(f"    median placebo slope:  {med_hc:+.4f}")
    print(f"    95% interval:          [{lo_hc:+.4f}, {hi_hc:+.4f}]")
    print(f"    observed HC slope:     {hc_obs_slope:+.4f}")
    print(f"    observed percentile:   {pct_hc:.1f}th")

    # ============================================================
    # OC PLACEBO
    # ============================================================
    print("\n" + "=" * 70)
    print("BLOCK (i) OC PLACEBO")
    print("=" * 70)
    oc_obs = pd.read_csv(ROOT / "oc_predictive_validation_paired.csv")
    oc_obs_intercept, oc_obs_slope, oc_obs_r2 = fit_ols(
        oc_obs["shift_pred"].values, oc_obs["shift_obs"].values)
    print(f"  OC observed slope: {oc_obs_slope:+.4f}  R²: {oc_obs_r2:.4f}  n: {len(oc_obs)}")

    print("  building RZ residuals (OC predictor base)...")
    rz = build_rz_residuals()
    print(f"  rz panel rows: {len(rz)}")

    cont_oc = continuous_staff_seasons(staff, "OC")
    print(f"  continuous-staff OC team-seasons candidates: {len(cont_oc)}")

    cont_rows_oc = []
    for _, r in cont_oc.iterrows():
        pre = compute_oc_profile(rz, r["team_nfl"], r["prior_season"], K, anchor="last")
        post = compute_oc_profile(rz, r["team_nfl"], r["new_season"], K, anchor="first")
        if not np.isnan(pre) and not np.isnan(post):
            cont_rows_oc.append({
                "team_nfl": r["team_nfl"], "new_season": r["new_season"],
                "profile_pre": pre, "profile_post": post,
                "shift_obs_placebo": post - pre,
            })
    cont_df_oc = pd.DataFrame(cont_rows_oc)
    print(f"  continuous-staff team-seasons with computable shift: {len(cont_df_oc)}")

    oc_pool_by_season = {}
    for s in sorted(oc_obs["new_season"].unique()):
        oc_pool_by_season[s] = oc_obs[oc_obs["new_season"] == s]["profile_oc_prior"].values
    broad_oc_pool = oc_obs["profile_oc_prior"].values

    placebo_slopes_oc = np.empty(B)
    for b in range(B):
        sampled = []
        for _, row in cont_df_oc.iterrows():
            pool = oc_pool_by_season.get(int(row["new_season"]), broad_oc_pool)
            if len(pool) == 0:
                pool = broad_oc_pool
            sampled.append(rng.choice(pool))
        sampled = np.array(sampled)
        shift_pred_placebo = sampled - cont_df_oc["profile_pre"].values
        _, slope_b, _ = fit_ols(shift_pred_placebo, cont_df_oc["shift_obs_placebo"].values)
        placebo_slopes_oc[b] = slope_b
    med_oc = float(np.median(placebo_slopes_oc))
    lo_oc, hi_oc = np.percentile(placebo_slopes_oc, [2.5, 97.5])
    pct_oc = float((placebo_slopes_oc < oc_obs_slope).mean() * 100.0)
    print(f"\n  OC PLACEBO RESULTS")
    print(f"    median placebo slope:  {med_oc:+.4f}")
    print(f"    95% interval:          [{lo_oc:+.4f}, {hi_oc:+.4f}]")
    print(f"    observed OC slope:     {oc_obs_slope:+.4f}")
    print(f"    observed percentile:   {pct_oc:.1f}th")

    # ============================================================
    # BLOCK (ii) — INFLUENCE: drop Reid PHI 2012 → KC 2013
    # ============================================================
    print("\n" + "=" * 70)
    print("BLOCK (ii) INFLUENCE: drop Reid PHI 2012 → KC 2013")
    print("=" * 70)
    mask = ~((hc_obs["HC"] == "Andy Reid")
              & (hc_obs["prior_team"] == "PHI")
              & (hc_obs["prior_season"] == 2012)
              & (hc_obs["new_team"] == "KC")
              & (hc_obs["new_season"] == 2013))
    n_dropped = (~mask).sum()
    print(f"  rows matching Reid PHI 2012 → KC 2013: {n_dropped}")
    hc_drop = hc_obs[mask].copy()
    x_d = hc_drop["shift_pred"].values
    y_d = hc_drop["shift_obs"].values
    intercept_d, slope_d, r2_d = fit_ols(x_d, y_d)
    r_d = float(np.corrcoef(x_d, y_d)[0, 1])
    p_d = perm_p(x_d, y_d, r_d, n_perm=5000, seed=42)
    print(f"  HC slope after drop:   {slope_d:+.4f}  (original {hc_obs_slope:+.4f})")
    print(f"  HC R² after drop:      {r2_d:.4f}   (original {hc_obs_r2:.4f})")
    print(f"  Pearson r after drop:  {r_d:+.4f}")
    print(f"  perm-p one-sided:      {p_d:.4f}")
    print(f"  n after drop:          {len(hc_drop)}")
    print(f"  SNR gate check (slope > 0 AND p < 0.05): "
          f"{'PASS' if (slope_d > 0 and p_d < 0.05) else 'FAIL'}")

    # ============================================================
    # Save artifacts
    # ============================================================
    out = pd.DataFrame({
        "metric": [
            "hc_placebo_median_slope",
            "hc_placebo_ci_lo",
            "hc_placebo_ci_hi",
            "hc_observed_slope",
            "hc_observed_pct",
            "oc_placebo_median_slope",
            "oc_placebo_ci_lo",
            "oc_placebo_ci_hi",
            "oc_observed_slope",
            "oc_observed_pct",
            "hc_drop_reid_slope",
            "hc_drop_reid_r2",
            "hc_drop_reid_r",
            "hc_drop_reid_pearson_perm_p",
            "hc_drop_reid_n",
        ],
        "value": [
            med_hc, lo_hc, hi_hc, hc_obs_slope, pct_hc,
            med_oc, lo_oc, hi_oc, oc_obs_slope, pct_oc,
            slope_d, r2_d, r_d, p_d, len(hc_drop),
        ],
    })
    out.to_csv(ROOT / "predictive_validation_robustness.csv", index=False)
    np.savez(ROOT / "predictive_validation_robustness_distributions.npz",
              placebo_hc=placebo_slopes_hc, placebo_oc=placebo_slopes_oc)
    print(f"\n  wrote: predictive_validation_robustness.csv + ...distributions.npz")

    # Final summary block for paper-ready quoting
    print("\n" + "=" * 70)
    print("PAPER-READY VALUES")
    print("=" * 70)
    print(f"\n(i) PLACEBO")
    print(f"  HC: median {med_hc:+.4f}, 95% CI [{lo_hc:+.4f}, {hi_hc:+.4f}], "
          f"observed {hc_obs_slope:+.2f} at {pct_hc:.1f}th percentile")
    print(f"  OC: median {med_oc:+.4f}, 95% CI [{lo_oc:+.4f}, {hi_oc:+.4f}], "
          f"observed {oc_obs_slope:+.2f} at {pct_oc:.1f}th percentile")
    print(f"\n(ii) INFLUENCE (drop Reid PHI 2012 → KC 2013)")
    print(f"  HC slope: {slope_d:+.4f}  R²: {r2_d:.4f}  p_perm: {p_d:.4f}  n: {len(hc_drop)}")
    print(f"  SNR gate: {'PASS' if (slope_d > 0 and p_d < 0.05) else 'FAIL'}")


if __name__ == "__main__":
    main()
