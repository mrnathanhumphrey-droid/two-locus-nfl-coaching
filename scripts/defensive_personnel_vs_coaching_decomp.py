"""Defensive personnel vs DC scheme — SVI-analogue variance decomposition.

Probe per Sloan §5 sharpening: defensive outcome residuals decomposed
between (a) defensive personnel composition (prior-season snap-weighted
draft capital) and (b) DC identity partition.

Reports three components per cell:
  unique(personnel | coaching)   — personnel info after partialling DC
  unique(coaching | personnel)   — DC info after partialling personnel
  overlap                         — joint authoring component

Honest framing: personnel and DC are endogenously correlated (DCs pick
players matching scheme; GMs hire DCs matching roster). The decomposition
quantifies joint authoring rather than forcing a single dominant-author
claim.

Operates on participation seasons 2016-2024 (defensive personnel coverage),
test-set seasons 2017-2024 (need prior-season snap shares).

Output:
  data/nfl/defensive_personnel_index.parquet
  data/nfl/defensive_outcome_panel.parquet
  data/nfl/defensive_svi_decomposition.csv
  docs/defensive_personnel_vs_coaching_summary.md
"""
from __future__ import annotations
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nfl"
PART_DIR = ROOT / "data" / "raw" / "participation"  # 2016-2024 full coverage
COACH_PATH = DATA / "coaching" / "coach_seasons.parquet"
DRAFT_PATH = DATA / "combine" / "draft_picks.parquet"
SR_PATH = DATA / "games" / "state_residuals.parquet"
PBP_DIR = ROOT / "data" / "nflfastR"

DEF_POSITIONS = {"DE", "DT", "NT", "DL",
                 "OLB", "ILB", "LB", "MLB",
                 "CB", "DB", "FS", "SS", "S",
                 # Roster files use coarser tags:
                 "LB", "DB"}


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def draft_value(pick):
    """Jimmy Johnson-style decay. Undrafted -> 0."""
    if pd.isna(pick) or pick <= 0:
        return 0.0
    p = float(pick)
    return 3000.0 / (1.0 + 0.3 * (p - 1)) ** 1.4


def build_player_snap_counts(years):
    """Per (team, season, gsis_id): defensive snap count from participation.

    Vectorized: explode defense_players, derive defteam from game_id format.
    Position is later joined from weekly_rosters (more reliable than the
    inconsistently-populated defense_positions column).
    """
    parts = []
    for y in years:
        p = PART_DIR / f"pbp_participation_{y}.parquet"
        if not p.exists():
            print(f"  participation missing for {y}, skip")
            continue
        df = pd.read_parquet(p, columns=["nflverse_game_id", "play_id",
                                          "possession_team", "defense_players"])
        df = df.dropna(subset=["defense_players", "possession_team"])
        gid = df["nflverse_game_id"].str.split("_")
        df["away"] = gid.str[2]
        df["home"] = gid.str[3]
        df["defteam"] = np.where(df["possession_team"] == df["home"],
                                  df["away"], df["home"])
        df["players_list"] = df["defense_players"].str.split(";")
        ex = df[["defteam", "players_list"]].explode("players_list")
        ex = ex.rename(columns={"players_list": "gsis_id", "defteam": "team"})
        ex = ex[ex["gsis_id"].notna() & (ex["gsis_id"] != "")]
        ex["season"] = y
        sc = ex.groupby(["season", "team", "gsis_id"]).size().reset_index(name="snaps")
        parts.append(sc)
        print(f"  {y}: {len(sc):,} unique (team, player) rows")
    if not parts:
        return pd.DataFrame(columns=["season", "team", "gsis_id", "position", "snaps"])
    snap_counts = pd.concat(parts, ignore_index=True)
    # Join position from weekly_rosters
    roster_parts = []
    for y in years:
        rp = ROOT / "data" / "nfl" / "rosters" / f"weekly_rosters_{y}.parquet"
        if not rp.exists():
            continue
        r = pd.read_parquet(rp, columns=["season", "gsis_id", "position"])
        r = r.drop_duplicates(subset=["season", "gsis_id"])
        roster_parts.append(r)
    if roster_parts:
        rosters = pd.concat(roster_parts, ignore_index=True)
        snap_counts = snap_counts.merge(rosters, on=["season", "gsis_id"], how="left")
        snap_counts["position"] = snap_counts["position"].fillna("UNK")
    else:
        snap_counts["position"] = "UNK"
    return snap_counts


def build_defensive_personnel_index(snap_counts, draft):
    """Per (team, season): snap-weighted draft-value index over defensive players.

    Filters to defensive positions to exclude offensive players accidentally
    in defense_players column.
    """
    snap_counts = snap_counts.copy()
    snap_counts["is_def"] = snap_counts["position"].isin(DEF_POSITIONS)
    snap_counts = snap_counts[snap_counts["is_def"]]
    # Total defensive snaps per (team, season) for normalization
    team_total = (snap_counts.groupby(["team", "season"])["snaps"].sum()
                                  .reset_index().rename(columns={"snaps": "team_def_snaps"}))
    snap_counts = snap_counts.merge(team_total, on=["team", "season"])
    snap_counts["snap_share"] = snap_counts["snaps"] / snap_counts["team_def_snaps"]

    # Draft lookup by gsis_id; use earliest pick if duplicated
    d = draft[["gsis_id", "pick"]].dropna(subset=["gsis_id"]).copy()
    d["pick"] = to_num(d["pick"])
    d = d.dropna(subset=["pick"]).sort_values("pick").drop_duplicates("gsis_id")
    d["draft_value"] = d["pick"].apply(draft_value)
    snap_counts = snap_counts.merge(d[["gsis_id", "pick", "draft_value"]], on="gsis_id", how="left")
    # Undrafted players: draft_value = 0
    snap_counts["draft_value"] = snap_counts["draft_value"].fillna(0.0)

    # Aggregate snap-weighted indices per (team, season)
    snap_counts["weighted_value"] = snap_counts["snap_share"] * snap_counts["draft_value"]
    idx = (snap_counts.groupby(["team", "season"])
                          .agg(personnel_index=("weighted_value", "sum"),
                               n_def_players=("gsis_id", "nunique"),
                               team_def_snaps=("team_def_snaps", "first"))
                          .reset_index())
    # Top-N starter indices for robustness
    snap_counts_sorted = snap_counts.sort_values(["team", "season", "snaps"],
                                                    ascending=[True, True, False])
    top11 = (snap_counts_sorted.groupby(["team", "season"]).head(11)
                  .groupby(["team", "season"])["draft_value"].mean()
                  .reset_index().rename(columns={"draft_value": "top11_mean_draft_value"}))
    top22 = (snap_counts_sorted.groupby(["team", "season"]).head(22)
                  .groupby(["team", "season"])["draft_value"].mean()
                  .reset_index().rename(columns={"draft_value": "top22_mean_draft_value"}))
    idx = idx.merge(top11, on=["team", "season"]).merge(top22, on=["team", "season"])
    return idx


def build_defensive_outcome_panel(years):
    """Per (team, season): defensive outcome aggregates.

    Uses state_residuals (offensive view per posteam) inverted to get the
    defensive view per defteam. Each game has one row per posteam; the
    defensive view averages over games where THIS team was the defteam.
    """
    sr = pd.read_parquet(SR_PATH)
    # Add defteam from game_id format
    gid = sr["game_id"].str.split("_")
    sr["away"] = gid.str[2]
    sr["home"] = gid.str[3]
    sr["defteam"] = np.where(sr["posteam"] == sr["home"], sr["away"], sr["home"])
    sr["season"] = to_num(sr["season"])
    sr = sr.dropna(subset=["season"])
    sr["season"] = sr["season"].astype(int)
    sr = sr[sr["season"].isin(years)]
    # Defensive aggregation: mean over games where defteam = this team
    agg = sr.groupby(["defteam", "season"]).agg(
        opp_pass_rate_residual=("pass_rate_residual", "mean"),
        opp_plays_residual=("plays_residual", "mean"),
        opp_rz_trips_residual=("red_zone_trips_residual", "mean"),
        opp_epa_per_play=("epa_per_play", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    # Add PBP-derived defensive RZ cells: opp_rz_pass_rate, opp_rz_sack_rate, opp_rz_td_rate
    rz_rows = []
    for y in years:
        for path in [PBP_DIR / f"play_by_play_{y}.parquet",
                     ROOT / "data" / "nfl" / "games" / f"pbp_{y}.parquet"]:
            if path.exists():
                pbp = pd.read_parquet(path, columns=[
                    "game_id", "season", "posteam", "defteam", "yardline_100",
                    "play_type", "pass", "rush", "sack", "touchdown", "season_type"])
                pbp = pbp[(pbp["season_type"] == "REG")
                          & (pbp["play_type"].isin(["pass", "run"]))
                          & (pbp["yardline_100"] <= 20)]
                rz = pbp.groupby(["defteam", "season"]).agg(
                    rz_plays=("pass", "size"),
                    rz_pass=("pass", "sum"),
                    rz_sack=("sack", "sum"),
                    rz_td=("touchdown", "sum"),
                ).reset_index().rename(columns={"defteam": "team"})
                rz["opp_rz_pass_rate"] = rz["rz_pass"] / rz["rz_plays"]
                rz["opp_rz_sack_rate"] = rz["rz_sack"] / rz["rz_plays"]
                rz["opp_rz_td_rate"] = rz["rz_td"] / rz["rz_plays"]
                rz_rows.append(rz[["team", "season", "opp_rz_pass_rate",
                                     "opp_rz_sack_rate", "opp_rz_td_rate", "rz_plays"]])
                break
    if rz_rows:
        rz_df = pd.concat(rz_rows, ignore_index=True)
        # League mean by season for residualization
        lm = rz_df.groupby("season")[["opp_rz_pass_rate", "opp_rz_sack_rate",
                                         "opp_rz_td_rate"]].mean().add_prefix("lg_").reset_index()
        rz_df = rz_df.merge(lm, on="season")
        for col in ["opp_rz_pass_rate", "opp_rz_sack_rate", "opp_rz_td_rate"]:
            rz_df[col + "_residual"] = rz_df[col] - rz_df["lg_" + col]
        rz_keep = rz_df[["team", "season",
                          "opp_rz_pass_rate_residual", "opp_rz_sack_rate_residual",
                          "opp_rz_td_rate_residual"]]
        agg = agg.merge(rz_keep, on=["team", "season"], how="left")
    return agg


def build_coaching_partition(years):
    """Per (team_pfr, season): HC + DC identity strings."""
    cs = pd.read_parquet(COACH_PATH)
    cs["games"] = cs["games"].fillna(0)
    # Pick primary coach per (team, season, role) by games
    primary = (cs[cs["role"].isin(["head_coach", "defensive_coordinator"])]
                  .sort_values(["team", "season", "role", "games"], ascending=[True, True, True, False])
                  .groupby(["team", "season", "role"]).first().reset_index())
    wide = primary.pivot_table(index=["team", "season"], columns="role",
                                  values="coach_name", aggfunc="first").reset_index()
    wide = wide.rename(columns={"head_coach": "hc", "defensive_coordinator": "dc"})
    wide["dc"] = wide["dc"].fillna("UNKNOWN_DC")
    wide["hc"] = wide["hc"].fillna("UNKNOWN_HC")
    wide["coaching_vector_id"] = wide["hc"].astype(str) + "|" + wide["dc"].astype(str)
    wide = wide[wide["season"].isin(years)].copy()
    return wide[["team", "season", "hc", "dc", "coaching_vector_id"]]


def r2(y, X):
    if X.shape[0] == 0 or X.shape[1] == 0:
        return 0.0
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


PFR_TO_NFL = {
    "rai": "LV", "rav": "BAL", "oti": "TEN", "sdg": "LAC", "htx": "HOU",
    "clt": "IND", "crd": "ARI", "ram": "LA", "buf": "BUF", "mia": "MIA",
    "nwe": "NE", "nyj": "NYJ", "cin": "CIN", "cle": "CLE", "pit": "PIT",
    "jax": "JAX", "den": "DEN", "kan": "KC", "dal": "DAL", "nyg": "NYG",
    "phi": "PHI", "was": "WAS", "chi": "CHI", "det": "DET", "gnb": "GB",
    "min": "MIN", "atl": "ATL", "car": "CAR", "nor": "NO", "tam": "TB",
    "sea": "SEA", "sfo": "SF",
}


def main():
    # Participation 2016-2024; tests run 2017-2024 (need prior-season).
    snap_years = list(range(2016, 2025))
    test_years = list(range(2017, 2025))

    print(f"=== building defensive personnel snap counts ({min(snap_years)}-{max(snap_years)}) ===")
    snap_counts = build_player_snap_counts(snap_years)
    print(f"  player-season-position rows: {len(snap_counts):,}")

    print("\n=== loading draft picks ===")
    draft = pd.read_parquet(DRAFT_PATH)
    print(f"  draft rows: {len(draft):,}  with gsis_id: {draft['gsis_id'].notna().sum():,}")

    print("\n=== building defensive personnel index per (team, season) ===")
    pidx = build_defensive_personnel_index(snap_counts, draft)
    print(f"  team-seasons: {len(pidx)}")
    print(f"  index summary:")
    print(pidx[["personnel_index", "top11_mean_draft_value",
                  "top22_mean_draft_value"]].describe().round(3).to_string())
    pidx.to_parquet(DATA / "defensive_personnel_index.parquet", index=False)

    print("\n=== building defensive outcome panel ===")
    outcomes = build_defensive_outcome_panel(test_years + [2016])  # include 2016 for completeness
    print(f"  outcome team-seasons: {len(outcomes)}")
    print(f"  cells available: {[c for c in outcomes.columns if c not in ('team', 'season')]}")
    outcomes.to_parquet(DATA / "defensive_outcome_panel.parquet", index=False)

    print("\n=== building coaching partition (HC + DC) ===")
    coach = build_coaching_partition(test_years)
    coach["team_nfl"] = coach["team"].map(PFR_TO_NFL).fillna(coach["team"])
    print(f"  coach rows: {len(coach)}  unique DCs: {coach['dc'].nunique()}  "
          f"unique HCs: {coach['hc'].nunique()}  unique coaching_vectors: "
          f"{coach['coaching_vector_id'].nunique()}")

    # Join: pidx (NFL codes) + outcomes (NFL codes) + coach (NFL codes via team_nfl)
    df = pidx.merge(outcomes, on=["team", "season"], how="inner")
    coach_clean = coach[["team_nfl", "season", "hc", "dc", "coaching_vector_id"]].copy()
    coach_clean = coach_clean.rename(columns={"team_nfl": "team"})
    df = df.merge(coach_clean, on=["team", "season"], how="inner")
    df = df[df["season"].isin(test_years)]
    print(f"\n  joined panel: {len(df)} team-seasons; "
          f"{df['dc'].nunique()} unique DCs; "
          f"{df['coaching_vector_id'].nunique()} unique coaching_vectors")
    df.to_parquet(DATA / "defensive_personnel_vs_coaching_joined.parquet", index=False)

    # ============================================================
    # SVI-analogue decomposition: outcomes ~ {personnel} vs {DC}
    # ============================================================
    print("\n=== SVI-analogue decomposition: defensive outcome ~ personnel vs DC ===")
    outcome_cols = [c for c in outcomes.columns if c not in ("team", "season")
                     and df[c].notna().sum() >= 100]
    print(f"  testing cells: {outcome_cols}")

    personnel_features = ["personnel_index", "top11_mean_draft_value", "top22_mean_draft_value"]
    print(f"  personnel features (Block A): {personnel_features}")

    dc_dummies = pd.get_dummies(df["dc"], drop_first=True).astype(float).values
    cv_dummies = pd.get_dummies(df["coaching_vector_id"], drop_first=True).astype(float).values
    print(f"  DC dummies (Block B_DC): {dc_dummies.shape[1]}")
    print(f"  coaching_vector dummies (Block B_CV): {cv_dummies.shape[1]}")

    rng = np.random.default_rng(2026)
    N_PERM = 200

    rows = []
    for partition_name, B in [("DC_only", dc_dummies),
                                 ("HC+DC_coaching_vector", cv_dummies)]:
        print(f"\n  ---- Block B partition: {partition_name} ({B.shape[1]} dummies, "
              f"perm null N={N_PERM}) ----")
        print(f"  {'cell':<28} {'R2_pers':>8} {'R2_coach':>9} {'null_mean':>10} "
              f"{'R2_coach_NET':>13} {'R2_both':>8} {'uniqA_NET':>10} {'uniqB_NET':>10} {'overlap':>8}")
        print(f"  {'-'*28} {'-'*8} {'-'*9} {'-'*10} {'-'*13} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for col in outcome_cols:
            y = df[col].values.astype(float)
            valid = ~np.isnan(y)
            y_v = y[valid]
            if len(y_v) < 50:
                continue
            X_pers = df[personnel_features].values[valid].astype(float)
            X_coach = B[valid]
            X_both = np.column_stack([X_pers, X_coach])
            r2_A = r2(y_v, X_pers)
            r2_B = r2(y_v, X_coach)
            r2_AB = r2(y_v, X_both)

            # Permutation null on the coaching dummies: shuffle ROW labels
            # (equivalently, shuffle which dummy-row goes to which y).
            null_B = []
            null_AB = []
            n = len(y_v)
            for _ in range(N_PERM):
                perm = rng.permutation(n)
                Bp = X_coach[perm]
                null_B.append(r2(y_v, Bp))
                null_AB.append(r2(y_v, np.column_stack([X_pers, Bp])))
            null_B_mean = float(np.mean(null_B))
            null_B_sd = float(np.std(null_B))
            null_AB_mean = float(np.mean(null_AB))

            # Net R² after subtracting permutation-expected R² (corrects high-dim bias)
            r2_B_net = r2_B - null_B_mean
            r2_AB_net = r2_AB - null_AB_mean
            uniq_A_net = r2_AB_net - r2_B_net  # personnel | coaching (NET)
            uniq_B_net = r2_AB_net - r2_A     # coaching | personnel (NET) — personnel is low-dim so no null adj
            overlap_net = r2_A + r2_B_net - r2_AB_net
            # Permutation p for coaching block
            p_coach = float(np.mean(np.array(null_B) >= r2_B))

            rows.append({
                "partition": partition_name,
                "cell": col,
                "n": int(valid.sum()),
                "r2_personnel": r2_A,
                "r2_coaching_raw": r2_B,
                "null_r2_coach_mean": null_B_mean,
                "null_r2_coach_sd": null_B_sd,
                "r2_coaching_NET": r2_B_net,
                "r2_both_raw": r2_AB,
                "r2_both_NET": r2_AB_net,
                "p_coach_perm": p_coach,
                "unique_personnel_given_coaching_NET": uniq_A_net,
                "unique_coaching_given_personnel_NET": uniq_B_net,
                "overlap_NET": overlap_net,
            })
            print(f"  {col:<28} {r2_A:>8.4f} {r2_B:>9.4f} {null_B_mean:>10.4f} "
                  f"{r2_B_net:>13.4f} {r2_AB:>8.4f} {uniq_A_net:>10.4f} {uniq_B_net:>10.4f} {overlap_net:>8.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "defensive_svi_decomposition.csv", index=False)
    print(f"\nwrote: defensive_svi_decomposition.csv ({len(out)} rows)")

    # Aggregate verdict per partition — NET values (permutation-bias-corrected)
    print("\n=== AGGREGATE VERDICT (permutation-null corrected) ===")
    for partition_name in ["DC_only", "HC+DC_coaching_vector"]:
        sub = out[out["partition"] == partition_name]
        if len(sub) == 0:
            continue
        med_uniq_p = sub["unique_personnel_given_coaching_NET"].median()
        med_uniq_c = sub["unique_coaching_given_personnel_NET"].median()
        med_overlap = sub["overlap_NET"].median()
        n_pers_wins = (sub["unique_personnel_given_coaching_NET"]
                       > sub["unique_coaching_given_personnel_NET"]).sum()
        # How many cells have coaching-block p_perm < 0.05?
        n_p_sig = (sub["p_coach_perm"] < 0.05).sum()
        print(f"\n  partition={partition_name}:")
        print(f"    median unique(personnel|coaching) NET = {med_uniq_p:+.4f}")
        print(f"    median unique(coaching|personnel) NET = {med_uniq_c:+.4f}")
        print(f"    median overlap                    NET = {med_overlap:+.4f}")
        print(f"    cells where personnel-unique > coaching-unique (NET): "
              f"{n_pers_wins}/{len(sub)}")
        print(f"    cells where coaching partition perm-p < 0.05: {n_p_sig}/{len(sub)}")


if __name__ == "__main__":
    main()
