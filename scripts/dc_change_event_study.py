"""DC-change event study — defensive symmetry of OC test.

Design parallel to OC-change study, but:
  - Treatment: offseason DC changes with HC continuity
  - Placebo: continuous staff (same HC AND same DC)
  - Targets: DEFENSIVE versions of the same metrics
    (opp_pass_rate_residual, opp_plays_residual, opp_rz_trips_residual)

The defensive metric for team T in a game = opponent's offensive metric
in that game (state_residuals indexed by posteam; T's defense is what
the opponent did with the ball on T).

If HC is the universal locus, DC test should null on opp pass_rate just
like OC test nulled on own pass_rate. If DC test carries, the HC-locus
claim is offense-side-specific.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_change_event_study import (
    PFR_TO_NFL, LEGACY_REMAP, K, N_PERM, permutation_test, compute_pre_post_delta,
)

ROOT = Path(__file__).resolve().parent.parent / "data"


def build_staff_panel_with_dc():
    """Pivot coach_seasons to (team_nfl, season) -> HC, DC names."""
    cs = pd.read_parquet(ROOT / "coaching/coach_seasons.parquet")
    cs["games_filled"] = cs["games"].fillna(0)
    primary = (cs[cs["role"].isin(["head_coach","defensive_coordinator"])]
               .sort_values(["team","season","role","games_filled"],
                            ascending=[True, True, True, False])
               .groupby(["team","season","role"]).first().reset_index())
    staff = primary.pivot_table(index=["team","season"], columns="role",
                                  values="coach_name", aggfunc="first").reset_index()
    staff = staff.rename(columns={"head_coach": "HC", "defensive_coordinator": "DC"})
    staff["team_nfl"] = staff["team"].map(PFR_TO_NFL)
    return staff[["team_nfl","season","HC","DC"]].dropna(subset=["team_nfl"])


def classify_dc_events(staff):
    """Identify DC-change-with-HC-continuity vs continuous-staff events."""
    staff = staff.copy().sort_values(["team_nfl","season"])
    staff["HC_prev"] = staff.groupby("team_nfl")["HC"].shift(1)
    staff["DC_prev"] = staff.groupby("team_nfl")["DC"].shift(1)
    staff["season_prev"] = staff.groupby("team_nfl")["season"].shift(1)
    s = staff[(staff["season"] - staff["season_prev"]) == 1].copy()
    s = s.dropna(subset=["HC","DC","HC_prev","DC_prev"])
    s["hc_same"] = s["HC"] == s["HC_prev"]
    s["dc_same"] = s["DC"] == s["DC_prev"]
    s["event_class"] = np.where(
        s["hc_same"] & ~s["dc_same"], "DC_CHANGE_HC_SAME",
        np.where(s["hc_same"] & s["dc_same"], "CONTINUOUS",
                 np.where(~s["hc_same"], "HC_CHANGE_EXCLUDE", "OTHER"))
    )
    return s


def build_defensive_residuals():
    """Build per-game DEFENSIVE residuals from state_residuals (offense rows).

    For each game, the row indexed by posteam=A describes A's offense.
    From defense-team B's perspective (where B is A's opponent), those same
    values are B's defense-allowed metrics for that game.
    """
    sr = pd.read_parquet(ROOT / "games/state_residuals.parquet")
    games = pd.read_parquet(ROOT / "games/games.parquet")
    if "week" not in sr.columns:
        sr = sr.merge(games[["game_id","week"]], on="game_id", how="left")

    # For each posteam row, find opponent team from games
    g_slim = games[["game_id","home_team","away_team","season","week"]].copy()
    sr2 = sr.merge(g_slim, on="game_id", how="left", suffixes=("","_g"))
    sr2["def_team"] = np.where(sr2["posteam"] == sr2["home_team"],
                                sr2["away_team"], sr2["home_team"])
    sr2["team_canon"] = sr2["def_team"].map(LEGACY_REMAP).fillna(sr2["def_team"])
    # Rename offensive metrics → defensive-allowed metrics
    sr2 = sr2.rename(columns={
        "pass_rate_residual": "opp_pass_rate_residual",
        "plays_residual": "opp_plays_residual",
        "red_zone_trips_residual": "opp_rz_trips_residual",
    })
    return sr2[["game_id","season","week","team_canon",
                 "opp_pass_rate_residual","opp_plays_residual","opp_rz_trips_residual"]]


def build_event_panel(staff_classified, sr_def, metrics):
    rows = []
    for _, ev in staff_classified.iterrows():
        if ev["event_class"] not in ("DC_CHANGE_HC_SAME", "CONTINUOUS"):
            continue
        tm, ssn = ev["team_nfl"], int(ev["season"])
        for met in metrics:
            d, n_prev, n_curr = compute_pre_post_delta(sr_def, tm, ssn, met)
            if np.isnan(d):
                continue
            rows.append({
                "team": tm, "season": ssn, "metric": met,
                "delta": d, "abs_delta": abs(d),
                "is_real_change": ev["event_class"] == "DC_CHANGE_HC_SAME",
                "HC": ev["HC"], "DC_prev": ev["DC_prev"], "DC_new": ev["DC"],
                "event_class": ev["event_class"],
            })
    return pd.DataFrame(rows)


def main():
    print(f"=== DC-CHANGE EVENT STUDY (K={K} games per window, n_perm={N_PERM}) ===\n")

    staff = build_staff_panel_with_dc()
    classified = classify_dc_events(staff)
    print(f"event class dist: {classified['event_class'].value_counts().to_dict()}")

    sr_def = build_defensive_residuals()
    print(f"defensive team-game rows: {len(sr_def):,}")

    metrics = ["opp_pass_rate_residual", "opp_plays_residual", "opp_rz_trips_residual"]
    panel = build_event_panel(classified, sr_def, metrics)

    n_real = panel[panel["is_real_change"]].groupby("metric").size().to_dict()
    n_plac = panel[~panel["is_real_change"]].groupby("metric").size().to_dict()
    print(f"\n  real DC-change events per metric: {n_real}")
    print(f"  placebo (continuous staff) events per metric: {n_plac}\n")

    print(f"  {'metric':<28} {'|d|_real':>10} {'|d|_plac':>10} {'ratio':>7} "
          f"{'p_perm':>8} {'verdict':>25}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*25}")
    summary = []
    for met in metrics:
        real = panel[(panel["metric"] == met) & panel["is_real_change"]]["abs_delta"].values
        plac = panel[(panel["metric"] == met) & ~panel["is_real_change"]]["abs_delta"].values
        if len(real) == 0 or len(plac) == 0:
            print(f"  {met:<28}  insufficient data")
            continue
        r_mean, p_mean = real.mean(), plac.mean()
        ratio = r_mean / p_mean if p_mean > 0 else np.nan
        obs_diff, p_val = permutation_test(real, plac)
        verdict = ("STAFF SHIFT MOVES NEEDLE" if p_val < 0.05 and ratio > 1.1
                   else "NO DETECTABLE SHIFT" if p_val >= 0.1 else "MARGINAL")
        summary.append({"metric": met, "n_real": len(real), "n_plac": len(plac),
                        "abs_d_real_mean": r_mean, "abs_d_plac_mean": p_mean,
                        "ratio": ratio, "p_perm": p_val, "verdict": verdict})
        print(f"  {met:<28} {r_mean:>10.4f} {p_mean:>10.4f} {ratio:>7.3f} "
              f"{p_val:>8.4f} {verdict:>25}")

    pd.DataFrame(summary).to_csv(ROOT / "dc_change_event_summary.csv", index=False)
    panel.to_csv(ROOT / "dc_change_event_panel.csv", index=False)
    print(f"\n  wrote {ROOT / 'dc_change_event_summary.csv'}")
    print(f"  wrote {ROOT / 'dc_change_event_panel.csv'}")

    print(f"\n=== Cross-reference: HC + OC results ===")
    print(f"  HC pass_rate:    1.36x, p=0.002  (STAFF SHIFT MOVES NEEDLE)")
    print(f"  OC pass_rate:    0.97x, p=0.636  (NO DETECTABLE SHIFT)")
    print(f"  OC RZ trips:     1.11x, p=0.075  (MARGINAL)")


if __name__ == "__main__":
    main()
