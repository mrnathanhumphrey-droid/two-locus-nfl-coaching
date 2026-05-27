"""Within-game SNR test (Layer 3) — score-state-segmented stylistic snapshots.

Hypothesis from brief:
  "Within-game: anticipated game state (script, matchups) vs actual
  (injuries, score state, weather) drives in-game style drift."

Test design:
  1. Segment pbp by score-differential bucket per (team, season):
       leading_2plus  : score_differential >= 9
       leading_1      : 1 <= score_differential <= 8
       tied           : score_differential == 0
       trailing_1     : -8 <= score_differential <= -1
       trailing_2plus : score_differential <= -9
  2. Per (team, season, bucket): compute stylistic features:
       pass_rate, shotgun_rate, deep_throw_rate, no_huddle_rate, run_pct
  3. Compare partition F-stat for coaching_vector vs team_id.
  4. "Style under stress" — for each style metric, compute the SHIFT:
       trailing_2plus_value − leading_2plus_value
     (how much does the team change tactics in different score states?)
     Test whether this SHIFT magnitude is partitioned more by coaching_vector
     than by team_id.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"


def f_stat(values, groups):
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if len(df) < 2:
        return np.nan, 0
    grand = df["v"].mean()
    grp_means = df.groupby("g")["v"].mean()
    grp_sizes = df.groupby("g")["v"].size()
    ng = len(grp_means)
    n = len(df)
    if ng < 2 or n - ng < 1:
        return np.nan, ng
    ss_b = float(((grp_means - grand) ** 2 * grp_sizes).sum())
    ss_w = float(sum(((df[df["g"] == g]["v"] - m) ** 2).sum()
                     for g, m in grp_means.items()))
    if ss_w <= 0:
        return np.nan, ng
    return (ss_b / (ng - 1)) / (ss_w / (n - ng)), ng


def score_bucket(diff):
    if pd.isna(diff):
        return "unknown"
    d = int(diff)
    if d >= 9:
        return "leading_2plus"
    elif d >= 1:
        return "leading_1"
    elif d == 0:
        return "tied"
    elif d >= -8:
        return "trailing_1"
    else:
        return "trailing_2plus"


def main():
    print("=== loading pbp ===")
    cols = ["game_id", "play_id", "season", "posteam", "play_type",
            "score_differential", "shotgun", "no_huddle", "qb_kneel",
            "qb_spike", "two_point_attempt", "pass_attempt", "rush_attempt",
            "air_yards", "down", "ydstogo"]
    pbp = pd.read_parquet(ROOT / "games" / "pbp.parquet", columns=cols)
    print(f"  {len(pbp):,} rows")

    pbp = pbp[(pbp["play_type"].isin(["pass", "run"]))
              & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)
              & (pbp["two_point_attempt"] != 1)
              & pbp["posteam"].notna()
              & pbp["score_differential"].notna()].copy()
    print(f"  filtered: {len(pbp):,} qualified plays")

    pbp["bucket"] = pbp["score_differential"].apply(score_bucket)
    pbp["is_pass"] = (pbp["play_type"] == "pass").astype(int)
    pbp["is_shotgun"] = pbp["shotgun"].fillna(0).astype(int)
    pbp["is_no_huddle"] = pbp["no_huddle"].fillna(0).astype(int)
    pbp["is_deep"] = ((pbp["air_yards"].fillna(0) >= 20)
                      & (pbp["play_type"] == "pass")).astype(int)

    print("\n=== aggregating per (team, season, bucket) ===")
    agg = (pbp.groupby(["posteam", "season", "bucket"]).agg(
        n_plays=("play_id", "count"),
        pass_rate=("is_pass", "mean"),
        shotgun_rate=("is_shotgun", "mean"),
        no_huddle_rate=("is_no_huddle", "mean"),
        deep_throw_rate=("is_deep", "mean"),
    ).reset_index())
    agg = agg[agg["n_plays"] >= 20]  # drop sparse cells
    print(f"  cells with >= 20 plays: {len(agg):,}")
    print(f"  per-bucket counts:")
    print(agg.groupby("bucket").size().to_string())

    # Attach coaching_vector_id
    cs = pd.read_parquet(ROOT / "coaching" / "coach_seasons.parquet")
    cs["games_filled"] = cs["games"].fillna(0)
    primary = (cs[cs["role"].isin(["head_coach", "offensive_coordinator", "defensive_coordinator"])]
               .sort_values(["team", "season", "role", "games_filled"], ascending=[True, True, True, False])
               .groupby(["team", "season", "role"]).first().reset_index())
    staff = primary.pivot_table(index=["team", "season"], columns="role",
                                 values="coach_name", aggfunc="first").reset_index()
    staff["coaching_vector_id"] = (staff.get("head_coach", "").fillna("") + "|"
                                    + staff.get("offensive_coordinator", "").fillna("") + "|"
                                    + staff.get("defensive_coordinator", "").fillna(""))
    pfr_to_nfl = {
        "rai": "LV", "rav": "BAL", "oti": "TEN", "sdg": "LAC", "htx": "HOU",
        "clt": "IND", "crd": "ARI", "ram": "LA", "buf": "BUF", "mia": "MIA",
        "nwe": "NE", "nyj": "NYJ", "cin": "CIN", "cle": "CLE", "pit": "PIT",
        "jax": "JAX", "den": "DEN", "kan": "KC", "dal": "DAL", "nyg": "NYG",
        "phi": "PHI", "was": "WAS", "chi": "CHI", "det": "DET", "gnb": "GB",
        "min": "MIN", "atl": "ATL", "car": "CAR", "nor": "NO", "tam": "TB",
        "sea": "SEA", "sfo": "SF",
    }
    legacy_remap = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA"}
    staff["team_nfl"] = staff["team"].map(pfr_to_nfl)
    agg["team_canon"] = agg["posteam"].map(legacy_remap).fillna(agg["posteam"])
    agg = agg.merge(staff[["team_nfl", "season", "coaching_vector_id"]],
                     left_on=["team_canon", "season"],
                     right_on=["team_nfl", "season"], how="left")
    agg = agg.dropna(subset=["coaching_vector_id"]).copy()
    print(f"  with coaching_vector attached: {len(agg):,}")

    style_cols = ["pass_rate", "shotgun_rate", "no_huddle_rate", "deep_throw_rate"]
    buckets = ["leading_2plus", "leading_1", "tied", "trailing_1", "trailing_2plus"]

    # ============== Test 1: per-bucket partition ==============
    print("\n=== TEST 1: per-bucket per-style partition F-stat ===")
    print(f"  hypothesis: in score-state X, does coaching_vector partition style better than team_id?")
    print()
    print(f"  {'bucket':<18} {'style':<18} {'n':>4} {'F_coach':>8} {'F_team':>8} "
          f"{'ratio':>8} {'verdict':>14}")
    print(f"  {'-'*18} {'-'*18} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*14}")
    rows1 = []
    for bucket in buckets:
        sub = agg[agg["bucket"] == bucket]
        for style in style_cols:
            F_c, ng_c = f_stat(sub[style].values, sub["coaching_vector_id"].values)
            F_t, ng_t = f_stat(sub[style].values, sub["team_canon"].values)
            ratio = F_c / F_t if F_t and F_t > 0 else np.nan
            verdict = ("COACH" if F_c > F_t * 1.5 and F_c > 1.5
                       else "TEAM" if F_t > F_c * 1.5 and F_t > 1.5
                       else "TIE")
            rows1.append({"bucket": bucket, "style": style, "n": len(sub),
                          "F_coach": F_c, "F_team": F_t, "ratio": ratio,
                          "verdict": verdict, "ng_coach": ng_c, "ng_team": ng_t})
            print(f"  {bucket:<18} {style:<18} {len(sub):>4} {F_c:>8.3f} {F_t:>8.3f} "
                  f"{ratio:>8.3f} {verdict:>14}")

    pd.DataFrame(rows1).to_csv(ROOT / "snr_within_game_per_bucket.csv", index=False)

    # ============== Test 2: stress shift ==============
    print("\n=== TEST 2: stress-shift partition (style under trailing vs leading) ===")
    print(f"  hypothesis: does coaching_vector explain how teams CHANGE style under pressure?")
    print()

    # Pivot to wide: per (team, season): style x bucket -> col
    wide = agg.pivot_table(index=["team_canon", "season", "coaching_vector_id"],
                            columns="bucket", values=style_cols).reset_index()
    # column names become tuples (style, bucket); flatten — but reset-index cols
    # have trailing empty strings, strip those
    wide.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c
                    for c in wide.columns]

    print(f"  {'shift':<35} {'n':>4} {'F_coach':>8} {'F_team':>8} "
          f"{'ratio':>8} {'verdict':>14}")
    print(f"  {'-'*35} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*14}")
    rows2 = []
    for style in style_cols:
        for bucket_a, bucket_b in [("trailing_2plus", "leading_2plus"),
                                    ("trailing_1", "leading_1"),
                                    ("trailing_2plus", "tied")]:
            col_a = f"{style}_{bucket_a}"
            col_b = f"{style}_{bucket_b}"
            if col_a not in wide.columns or col_b not in wide.columns:
                continue
            shift = wide[col_a] - wide[col_b]
            valid = shift.notna()
            label = f"{style}: {bucket_a}-{bucket_b}"
            F_c, ng_c = f_stat(shift[valid].values, wide.loc[valid, "coaching_vector_id"].values)
            F_t, ng_t = f_stat(shift[valid].values, wide.loc[valid, "team_canon"].values)
            ratio = F_c / F_t if F_t and F_t > 0 else np.nan
            verdict = ("COACH" if F_c > F_t * 1.5 and F_c > 1.5
                       else "TEAM" if F_t > F_c * 1.5 and F_t > 1.5
                       else "TIE")
            rows2.append({"shift": label, "n": int(valid.sum()),
                          "F_coach": F_c, "F_team": F_t, "ratio": ratio,
                          "verdict": verdict})
            print(f"  {label:<35} {int(valid.sum()):>4} {F_c:>8.3f} {F_t:>8.3f} "
                  f"{ratio:>8.3f} {verdict:>14}")

    pd.DataFrame(rows2).to_csv(ROOT / "snr_within_game_stress_shift.csv", index=False)

    # ============== Headline ==============
    print("\n=== HEADLINE ===")
    df1 = pd.DataFrame(rows1)
    df2 = pd.DataFrame(rows2)
    coach_wins_t1 = (df1["verdict"] == "COACH").sum()
    team_wins_t1 = (df1["verdict"] == "TEAM").sum()
    ties_t1 = (df1["verdict"] == "TIE").sum()
    coach_wins_t2 = (df2["verdict"] == "COACH").sum()
    team_wins_t2 = (df2["verdict"] == "TEAM").sum()
    ties_t2 = (df2["verdict"] == "TIE").sum()
    print(f"  Test 1 (per-bucket partition): COACH {coach_wins_t1} / TEAM {team_wins_t1} / TIE {ties_t1} (of {len(df1)})")
    print(f"  Test 2 (stress-shift partition): COACH {coach_wins_t2} / TEAM {team_wins_t2} / TIE {ties_t2} (of {len(df2)})")


if __name__ == "__main__":
    main()
