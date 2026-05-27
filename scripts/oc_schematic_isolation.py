"""Layer 3 schematic isolation — does the within-game COACH dominance live
at HC, OC, or DC level?

Original Layer 3 used coaching_vector_id = HC|OC|DC composite. Found
COACH 15/20 cells on shotgun/no-huddle/deep-throw. We don't know which
of the three roles drives that.

This test runs F-stat partitions by each role independently, plus the
composite, on the same per-(team, season, bucket) aggregate panel.

For each (bucket × style) cell:
  F_HC      — partition by HC_id only
  F_OC      — partition by OC_id only
  F_DC      — partition by DC_id only
  F_HC_OC   — partition by HC|OC composite
  F_team    — partition by team_canon (baseline)
  F_combined — coaching_vector_id (HC|OC|DC), original Layer 3 partition

Question: which role's partition has highest F? Per role, also report
within-season permutation null + p-value.

If HC dominates → schematic identity is HEAD-COACH-driven.
If OC dominates → schematic identity is OC-driven (OC nulls on outcomes,
                  positive on schematic = OC owns formation/tempo, not magnitude)
If composite > sum of parts → joint identity matters
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_within_game_placebo_audit import (
    f_stat, f_only, permute_within_strata, score_bucket,
)

ROOT = Path(__file__).resolve().parent.parent / "data"
N_PERM = 1000
RNG_SEED = 42


def build_agg_with_roles():
    """Same per-(team, season, bucket) aggregate but keep HC/OC/DC names separately."""
    cols = ["game_id", "play_id", "season", "posteam", "play_type",
            "score_differential", "shotgun", "no_huddle", "qb_kneel",
            "qb_spike", "two_point_attempt", "air_yards"]
    pbp = pd.read_parquet(ROOT / "games" / "pbp.parquet", columns=cols)
    pbp = pbp[(pbp["play_type"].isin(["pass","run"]))
              & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)
              & (pbp["two_point_attempt"] != 1)
              & pbp["posteam"].notna()
              & pbp["score_differential"].notna()].copy()
    pbp["bucket"] = pbp["score_differential"].apply(score_bucket)
    pbp["is_pass"] = (pbp["play_type"] == "pass").astype(int)
    pbp["is_shotgun"] = pbp["shotgun"].fillna(0).astype(int)
    pbp["is_no_huddle"] = pbp["no_huddle"].fillna(0).astype(int)
    pbp["is_deep"] = ((pbp["air_yards"].fillna(0) >= 20)
                      & (pbp["play_type"] == "pass")).astype(int)
    agg = (pbp.groupby(["posteam","season","bucket"]).agg(
        n_plays=("play_id", "count"),
        pass_rate=("is_pass", "mean"),
        shotgun_rate=("is_shotgun", "mean"),
        no_huddle_rate=("is_no_huddle", "mean"),
        deep_throw_rate=("is_deep", "mean"),
    ).reset_index())
    agg = agg[agg["n_plays"] >= 20]

    cs = pd.read_parquet(ROOT / "coaching" / "coach_seasons.parquet")
    cs["games_filled"] = cs["games"].fillna(0)
    primary = (cs[cs["role"].isin(["head_coach","offensive_coordinator","defensive_coordinator"])]
               .sort_values(["team","season","role","games_filled"], ascending=[True,True,True,False])
               .groupby(["team","season","role"]).first().reset_index())
    staff = primary.pivot_table(index=["team","season"], columns="role",
                                 values="coach_name", aggfunc="first").reset_index()
    staff = staff.rename(columns={"head_coach":"HC","offensive_coordinator":"OC","defensive_coordinator":"DC"})
    staff["HC_OC"] = staff["HC"].fillna("") + "|" + staff["OC"].fillna("")
    staff["coaching_vector_id"] = (staff["HC"].fillna("") + "|"
                                    + staff["OC"].fillna("") + "|"
                                    + staff["DC"].fillna(""))
    pfr_to_nfl = {
        "rai":"LV","rav":"BAL","oti":"TEN","sdg":"LAC","htx":"HOU","clt":"IND",
        "crd":"ARI","ram":"LA","buf":"BUF","mia":"MIA","nwe":"NE","nyj":"NYJ",
        "cin":"CIN","cle":"CLE","pit":"PIT","jax":"JAX","den":"DEN","kan":"KC",
        "dal":"DAL","nyg":"NYG","phi":"PHI","was":"WAS","chi":"CHI","det":"DET",
        "gnb":"GB","min":"MIN","atl":"ATL","car":"CAR","nor":"NO","tam":"TB",
        "sea":"SEA","sfo":"SF",
    }
    legacy_remap = {"STL":"LA","SD":"LAC","OAK":"LV","LAR":"LA"}
    staff["team_nfl"] = staff["team"].map(pfr_to_nfl)
    agg["team_canon"] = agg["posteam"].map(legacy_remap).fillna(agg["posteam"])
    agg = agg.merge(staff[["team_nfl","season","HC","OC","DC","HC_OC","coaching_vector_id"]],
                     left_on=["team_canon","season"],
                     right_on=["team_nfl","season"], how="left")
    agg = agg.dropna(subset=["coaching_vector_id"]).copy()
    # Fill any remaining NaN role names with empty string for partition stability
    for col in ["HC","OC","DC","HC_OC"]:
        agg[col] = agg[col].fillna("").astype(str)
    return agg


def main():
    print("=== building schematic aggregate panel ===")
    agg = build_agg_with_roles()
    print(f"  {len(agg):,} rows")

    style_cols = ["pass_rate","shotgun_rate","no_huddle_rate","deep_throw_rate"]
    buckets = ["leading_2plus","leading_1","tied","trailing_1","trailing_2plus"]
    role_partitions = [
        ("HC", "HC"),
        ("OC", "OC"),
        ("DC", "DC"),
        ("HC_OC", "HC_OC"),
        ("coach_full", "coaching_vector_id"),
        ("team_id", "team_canon"),
    ]

    rng = np.random.default_rng(RNG_SEED)
    print(f"\n=== role-level F-stat decomposition (n_perm={N_PERM}) ===\n")
    print(f"  {'bucket':<16} {'style':<18} "
          + " ".join(f"{name:>10}" for name, _ in role_partitions))
    print(f"  {'-'*16} {'-'*18} " + " ".join('-'*10 for _ in role_partitions))

    rows = []
    for bucket in buckets:
        sub = agg[agg["bucket"] == bucket].reset_index(drop=True)
        if len(sub) < 4:
            continue
        for style in style_cols:
            y = sub[style].values
            line = f"  {bucket:<16} {style:<18} "
            fmap = {}
            for name, col in role_partitions:
                if col not in sub.columns:
                    fmap[name] = (np.nan, 0)
                    line += f"{'-':>10} "
                    continue
                F, ng = f_stat(y, sub[col].values)
                fmap[name] = (F, ng)
                line += f"{F:>10.2f}"
            print(line)
            # Save row
            r = {"bucket": bucket, "style": style, "n": len(sub)}
            for name, _ in role_partitions:
                r[f"F_{name}"] = fmap[name][0]
                r[f"ng_{name}"] = fmap[name][1]
            rows.append(r)

    df = pd.DataFrame(rows)
    out = ROOT / "oc_schematic_isolation_results.csv"
    df.to_csv(out, index=False)
    print(f"\n  wrote {out}")

    # Summary: which role wins per cell (highest F among HC, OC, DC)
    print(f"\n=== which role has highest F per cell? ===")
    df["winner"] = df[["F_HC","F_OC","F_DC"]].idxmax(axis=1).str.replace("F_","")
    print(df[["bucket","style","F_HC","F_OC","F_DC","winner"]].to_string(index=False))
    print(f"\n  winner counts: {df['winner'].value_counts().to_dict()}")

    # Mean F per role across style cols
    print(f"\n=== mean F across all 20 cells by role ===")
    role_means = {name: df[f"F_{name}"].mean() for name, _ in role_partitions}
    for name, v in sorted(role_means.items(), key=lambda x: -x[1]):
        print(f"  {name:<14}  mean F = {v:.2f}")

    # By style: which role wins
    print(f"\n=== winner by schematic style (across 5 buckets each) ===")
    for style in style_cols:
        s = df[df["style"] == style]
        print(f"  {style:<18}  winner_counts = {s['winner'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
