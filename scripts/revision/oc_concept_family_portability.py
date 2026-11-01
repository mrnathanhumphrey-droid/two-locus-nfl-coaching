"""Concept-family stratification of OC predictive-validation slopes.

Paper §6.2 predicts:
  Roster-independent (field-position-conditioned) → HIGH portability slope
    - RZ shotgun rate (paper's headline, slope ~0.45-0.47)
    - Goal-line shotgun rate (yardline_100 <= 5)
    - 2-min no-huddle rate (game_seconds_remaining <= 120)
  Roster-conditioned → LOW portability slope
    - Deep-shot rate (air_yards >= 20 / pass attempts; 2006+ only)
    - Overall (non-RZ) shotgun rate
    - Early-down neutral pass rate

Build per-(game, posteam) panel for all concepts, residualize against league mean per
season, then run the OC predictive-validation pipeline per concept. Compare slopes.

Note: play_action and RPO frequency (named in paper) are NOT in this PBP dataset.
Reported as data limitations.
"""
from __future__ import annotations
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from oc_change_event_study import LEGACY_REMAP, build_staff_panel
from hc_predictive_validation import permutation_test_corr, fit_ols, N_PERM_TEST, K
from oc_predictive_validation import find_retread_oc_events

DATA = Path(__file__).resolve().parents[2] / "data" / "nfl"
OUT = DATA / "sloan_revision_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def build_concept_panel() -> pd.DataFrame:
    """Per (game, posteam, season, week) compute all concept rates."""
    cols = ["game_id", "play_id", "season", "week", "posteam", "play_type",
            "yardline_100", "shotgun", "no_huddle", "qb_kneel", "qb_spike",
            "two_point_attempt", "qtr", "game_seconds_remaining", "half_seconds_remaining",
            "down", "ydstogo", "air_yards", "pass", "season_type", "score_differential"]
    print("Loading PBP...", flush=True)
    pbp = pd.read_parquet(DATA / "games" / "pbp.parquet", columns=cols)
    pbp = pbp[(pbp["season_type"] == "REG")
              & (pbp["play_type"].isin(["pass", "run"]))
              & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)
              & (pbp["two_point_attempt"] != 1)
              & pbp["posteam"].notna()].copy()
    pbp["is_pass"] = (pbp["play_type"] == "pass").astype(int)
    pbp["is_shotgun"] = pbp["shotgun"].fillna(0).astype(int)
    pbp["is_no_huddle"] = pbp["no_huddle"].fillna(0).astype(int)
    pbp["is_rz"] = (pbp["yardline_100"] <= 20).astype(int)
    pbp["is_goalline"] = (pbp["yardline_100"] <= 5).astype(int)
    pbp["is_neutral"] = (pbp["score_differential"].abs() <= 8).astype(int)
    pbp["is_early_down"] = pbp["down"].isin([1, 2]).astype(int)
    pbp["is_late_2min"] = (pbp["half_seconds_remaining"] <= 120).astype(int)
    pbp["is_deep"] = (pbp["air_yards"] >= 20).astype(int)

    print(f"  pbp rows post-filter: {len(pbp):,}", flush=True)

    print("Computing per-(game, posteam) concept rates...", flush=True)
    g = pbp.groupby(["game_id", "season", "week", "posteam"], as_index=False)

    def safe_rate(num, den):
        return num.sum() / den.sum() if den.sum() > 0 else np.nan

    # RZ shotgun rate (already paper's metric)
    rows = []
    for (gid, s, w, pos), sub in g:
        rz_plays = sub["is_rz"]
        gl_plays = sub["is_goalline"]
        pass_plays = sub["is_pass"]
        late2_plays = sub["is_late_2min"]
        early_neutral = (sub["is_early_down"] == 1) & (sub["is_neutral"] == 1) & (sub["qtr"] <= 3)
        rows.append({
            "game_id": gid, "season": s, "week": w, "posteam": pos,
            "n_plays": len(sub),
            "rz_n": int(rz_plays.sum()),
            "rz_shotgun_rate": float(sub.loc[rz_plays == 1, "is_shotgun"].mean()) if rz_plays.sum() > 0 else np.nan,
            "gl_n": int(gl_plays.sum()),
            "goalline_shotgun_rate": float(sub.loc[gl_plays == 1, "is_shotgun"].mean()) if gl_plays.sum() >= 2 else np.nan,
            "late2_n": int(late2_plays.sum()),
            "late2_no_huddle_rate": float(sub.loc[late2_plays == 1, "is_no_huddle"].mean()) if late2_plays.sum() >= 3 else np.nan,
            "non_rz_n": int((sub["is_rz"] == 0).sum()),
            "non_rz_shotgun_rate": float(sub.loc[sub["is_rz"] == 0, "is_shotgun"].mean()) if (sub["is_rz"] == 0).sum() > 0 else np.nan,
            "pass_n": int(pass_plays.sum()),
            "deep_n": int(sub.loc[pass_plays == 1, "is_deep"].sum()),
            "deep_shot_rate": float(sub.loc[(pass_plays == 1) & (sub["air_yards"].notna()), "is_deep"].mean()) if (pass_plays == 1).sum() >= 5 else np.nan,
            "early_neutral_n": int(early_neutral.sum()),
            "early_neutral_pass_rate": float(sub.loc[early_neutral, "is_pass"].mean()) if early_neutral.sum() >= 5 else np.nan,
        })
    panel = pd.DataFrame(rows)
    print(f"  panel rows: {len(panel):,}", flush=True)

    # Residualize each concept against league mean per season
    concepts = ["rz_shotgun_rate", "goalline_shotgun_rate", "late2_no_huddle_rate",
                  "non_rz_shotgun_rate", "deep_shot_rate", "early_neutral_pass_rate"]
    for c in concepts:
        lm = panel.groupby("season")[c].transform("mean")
        panel[f"{c}_residual"] = panel[c] - lm

    # Add team_canon for the staff-panel join
    panel["team_canon"] = panel["posteam"].map(LEGACY_REMAP).fillna(panel["posteam"])
    return panel


CONCEPTS = [
    ("rz_shotgun_rate", "Roster-INDEP", "RZ shotgun rate (paper headline)"),
    ("goalline_shotgun_rate", "Roster-INDEP", "Goal-line shotgun rate (<=5 yd)"),
    ("late2_no_huddle_rate", "Roster-INDEP", "Late-2-min no-huddle rate"),
    ("non_rz_shotgun_rate", "Mixed", "Non-RZ shotgun rate (overall)"),
    ("deep_shot_rate", "Roster-COND", "Deep-shot rate (air_yds>=20, 2006+)"),
    ("early_neutral_pass_rate", "Roster-COND", "Early-down neutral pass rate"),
]


def compute_profile(panel: pd.DataFrame, team: str, season: int, k: int, anchor: str, concept_col: str) -> float:
    g = panel[(panel["team_canon"] == team) & (panel["season"] == season)].sort_values("week")
    g = g.dropna(subset=[concept_col])
    if len(g) < k:
        return np.nan
    sub = g.iloc[-k:] if anchor == "last" else g.iloc[:k]
    return float(sub[concept_col].mean())


def run_predictive_per_concept(events: pd.DataFrame, panel: pd.DataFrame, concept_col: str) -> pd.DataFrame:
    rows = []
    for _, ev in events.iterrows():
        oc = ev["OC"]
        prior_team, prior_season = ev["prior_team"], ev["prior_season"]
        new_team, new_season = ev["new_team"], ev["new_season"]
        if pd.isna(prior_team) or pd.isna(new_team):
            continue
        p_prior = compute_profile(panel, prior_team, prior_season, K, "last", concept_col)
        p_pre = compute_profile(panel, new_team, new_season - 1, K, "last", concept_col)
        p_post = compute_profile(panel, new_team, new_season, K, "first", concept_col)
        if any(np.isnan([p_prior, p_pre, p_post])):
            continue
        rows.append({
            "OC": oc, "prior_team": prior_team, "prior_season": prior_season,
            "new_team": new_team, "new_season": new_season,
            "profile_oc_prior": p_prior, "profile_new_pre": p_pre, "profile_new_post": p_post,
            "shift_obs": p_post - p_pre, "shift_pred": p_prior - p_pre,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 80, flush=True)
    print("Sloan Revision #3 — Concept-family stratification of OC portability", flush=True)
    print("=" * 80, flush=True)

    panel = build_concept_panel()

    staff = build_staff_panel()
    events = find_retread_oc_events(staff)
    print(f"\nRetread OC events: {len(events)}", flush=True)

    results = []
    for concept_col, bucket, name in CONCEPTS:
        residual_col = f"{concept_col}_residual"
        if residual_col not in panel.columns:
            print(f"\n  {name}: column missing", flush=True); continue
        paired = run_predictive_per_concept(events, panel, residual_col)
        if len(paired) < 10:
            print(f"\n  {name}: only {len(paired)} events with computable profiles — skipping", flush=True)
            continue

        x = paired["shift_pred"].values
        y = paired["shift_obs"].values
        obs_r, p_one, p_two, null_rs = permutation_test_corr(x, y)
        intercept, slope, r2 = fit_ols(x, y)
        agree = ((paired["shift_pred"] > 0) == (paired["shift_obs"] > 0)).sum()
        n = len(paired)

        results.append({
            "concept": concept_col, "bucket": bucket, "name": name,
            "n_events": n, "pearson_r": obs_r, "p_one_sided": p_one,
            "ols_slope": slope, "ols_r2": r2, "dir_agreement": float(agree / n),
        })
        print(f"\n  --- {bucket}: {name} ---", flush=True)
        print(f"    n_events: {n}", flush=True)
        print(f"    Pearson r: {obs_r:+.4f}  p_one={p_one:.4f}", flush=True)
        print(f"    OLS slope: {slope:+.4f}  R²: {r2:.4f}", flush=True)
        print(f"    Direction agreement: {agree}/{n} ({100 * agree / n:.0f}%)", flush=True)

        # Save per-concept paired
        paired.to_csv(OUT / f"oc_paired_{concept_col}.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "oc_concept_family_results.csv", index=False)

    print("\n" + "=" * 100, flush=True)
    print("CONCEPT-FAMILY COMPARISON — OC PORTABILITY", flush=True)
    print("=" * 100, flush=True)
    print(f"  Paper §6.2 prediction: Roster-INDEP slopes > Roster-COND slopes", flush=True)
    print()
    for bucket in ["Roster-INDEP", "Mixed", "Roster-COND"]:
        sub = results_df[results_df["bucket"] == bucket]
        if not len(sub):
            continue
        print(f"\n  {bucket}:", flush=True)
        for _, row in sub.iterrows():
            sig = " (SIG)" if row["p_one_sided"] < 0.05 else ""
            print(f"    {row['name']:42s}  n={int(row['n_events']):3d}  r={row['pearson_r']:+.3f}  slope={row['ols_slope']:+.3f}  R²={row['ols_r2']:.3f}  p={row['p_one_sided']:.3f}{sig}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("PAPER §6.2 PREDICTION TEST", flush=True)
    print("=" * 100, flush=True)
    indep_slopes = results_df[results_df["bucket"] == "Roster-INDEP"]["ols_slope"].values
    cond_slopes = results_df[results_df["bucket"] == "Roster-COND"]["ols_slope"].values
    print(f"  Roster-INDEP slopes: mean={indep_slopes.mean():.3f}  (range {indep_slopes.min():.3f}-{indep_slopes.max():.3f})", flush=True)
    print(f"  Roster-COND slopes:  mean={cond_slopes.mean():.3f}  (range {cond_slopes.min():.3f}-{cond_slopes.max():.3f})", flush=True)
    print(f"  Prediction: INDEP > COND.  Observed: INDEP={indep_slopes.mean():.3f} {' > ' if indep_slopes.mean() > cond_slopes.mean() else ' < '} COND={cond_slopes.mean():.3f}", flush=True)

    # Save concept panel
    panel.to_parquet(OUT / "oc_concept_panel.parquet")
    print(f"\nSaved concept panel -> {OUT / 'oc_concept_panel.parquet'}", flush=True)
    print(f"Saved results -> {OUT / 'oc_concept_family_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
