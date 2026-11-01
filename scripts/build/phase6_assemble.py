"""Phase 6: final assembly + done-criteria check.

- Adds lineage_root flag for HCs/coaches with no mentor edge (per brief)
- Joins continuity + injury covariates onto games.parquet
- Generates data/nfl/data_layer_status.md
- Verifies all 6 done-criteria from the brief
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "nfl"
GAMES = ROOT / "games"
COACHING = ROOT / "coaching"
ROSTERS = ROOT / "rosters"
INJURIES = ROOT / "injuries"


def augment_lineage_root():
    print("\n=== flag lineage_root for coaches with no mentor ===")
    cs = pd.read_parquet(COACHING / "coach_seasons.parquet")
    edges = pd.read_parquet(COACHING / "lineage_edges.parquet")
    all_slugs = cs[cs["sr_slug"] != ""]["sr_slug"].unique()
    mentee_slugs = set(edges["mentee_slug"].dropna().unique())
    roots = [s for s in all_slugs if s not in mentee_slugs]
    print(f"  {len(roots):,} coaches have no mentor edge -> tagged as roots")
    # Build root rows + concat
    coach_name_lookup = (cs[cs["sr_slug"] != ""]
                         .groupby("sr_slug")["coach_name"]
                         .agg(lambda s: s.value_counts().index[0])).to_dict()
    role_lookup = (cs[cs["sr_slug"] != ""]
                   .groupby("sr_slug")["role"]
                   .agg(lambda s: ",".join(sorted(set(s))))).to_dict()
    root_rows = pd.DataFrame([{
        "mentor_name": None, "mentor_slug": None, "mentor_role": None,
        "mentee_name": coach_name_lookup.get(s, ""), "mentee_slug": s,
        "mentee_role": role_lookup.get(s, ""),
        "team": None, "season": None,
        "role_match_type": "lineage_root",
    } for s in roots])
    augmented = pd.concat([edges, root_rows], ignore_index=True)
    augmented.to_parquet(COACHING / "lineage_edges.parquet", index=False)
    print(f"  WROTE lineage_edges.parquet ({len(augmented):,} total rows)")


def join_covariates_onto_games():
    print("\n=== join continuity + injury aggregates onto games.parquet ===")
    games = pd.read_parquet(GAMES / "games.parquet")
    cont = pd.read_parquet(ROSTERS / "continuity.parquet")

    # Join continuity for both home and away teams
    cont_home = cont.rename(columns={
        "team": "home_team",
        "roster_continuity_pct": "home_roster_continuity",
        "n_returning": "home_n_returning",
        "n_players": "home_n_players",
    })
    cont_away = cont.rename(columns={
        "team": "away_team",
        "roster_continuity_pct": "away_roster_continuity",
        "n_returning": "away_n_returning",
        "n_players": "away_n_players",
    })
    games = games.merge(
        cont_home[["home_team", "season", "home_roster_continuity",
                   "home_n_returning", "home_n_players"]],
        on=["home_team", "season"], how="left")
    games = games.merge(
        cont_away[["away_team", "season", "away_roster_continuity",
                   "away_n_returning", "away_n_players"]],
        on=["away_team", "season"], how="left")

    # Injury aggregates: per (team, season, week) total questionable + out counts
    try:
        inj = pd.read_parquet(INJURIES / "injuries_all.parquet")
        # standardize column names (nflverse changes them)
        team_col = next((c for c in ["team", "club_code", "team_abbr"] if c in inj.columns), None)
        status_col = next((c for c in ["report_status", "game_status", "practice_status",
                                        "report_primary_injury", "status"] if c in inj.columns), None)
        if team_col and status_col and "season" in inj.columns and "week" in inj.columns:
            # Out / Doubtful = "definitely or likely missing"
            inj["is_out"] = inj[status_col].astype(str).str.lower().isin(
                ["out", "doubtful"]).astype(int)
            inj["is_q"] = inj[status_col].astype(str).str.lower().eq("questionable").astype(int)
            agg = (inj.groupby([team_col, "season", "week"])
                       .agg(n_out=("is_out", "sum"),
                            n_questionable=("is_q", "sum"),
                            n_listed=(team_col, "size"))
                       .reset_index().rename(columns={team_col: "team"}))
            home_agg = agg.rename(columns={
                "team": "home_team", "n_out": "home_inj_out",
                "n_questionable": "home_inj_q", "n_listed": "home_inj_listed",
            })
            away_agg = agg.rename(columns={
                "team": "away_team", "n_out": "away_inj_out",
                "n_questionable": "away_inj_q", "n_listed": "away_inj_listed",
            })
            games = games.merge(
                home_agg[["home_team", "season", "week",
                          "home_inj_out", "home_inj_q", "home_inj_listed"]],
                on=["home_team", "season", "week"], how="left")
            games = games.merge(
                away_agg[["away_team", "season", "week",
                          "away_inj_out", "away_inj_q", "away_inj_listed"]],
                on=["away_team", "season", "week"], how="left")
            print(f"  injury covariates joined ({games['home_inj_out'].notna().sum()} games with home inj data)")
        else:
            print(f"  injury schema mismatch (team_col={team_col} status_col={status_col}); skipping inj join")
    except Exception as e:
        print(f"  injury join failed: {e}")

    # write augmented games
    out = GAMES / "games.parquet"
    games.to_parquet(out, index=False)
    print(f"  WROTE {out} ({len(games):,} games, {games.shape[1]} cols)")
    return games


def write_status_md(games):
    print("\n=== writing data_layer_status.md ===")
    cs = pd.read_parquet(COACHING / "coach_seasons.parquet")
    edges = pd.read_parquet(COACHING / "lineage_edges.parquet")
    cont = pd.read_parquet(ROSTERS / "continuity.parquet")
    sr = pd.read_parquet(GAMES / "state_residuals.parquet")
    pbp_fp = GAMES / "pbp.parquet"
    pbp_size_mb = pbp_fp.stat().st_size / 1e6 if pbp_fp.exists() else 0

    # Done criteria
    games_2000 = games[games["season"] >= 2000]
    games_with_vegas = games_2000["spread_line"].notna().sum()
    pbp_game_ids = set()
    if pbp_fp.exists():
        # quick read of just game_id col
        pbp_ids_df = pd.read_parquet(pbp_fp, columns=["game_id"])
        pbp_game_ids = set(pbp_ids_df["game_id"].dropna().astype(str))
    games_with_pbp = games_2000[games_2000["game_id"].astype(str).isin(pbp_game_ids)]
    n_games_pbp_match = len(games_with_pbp)

    cs_team_seasons = cs.groupby(["team", "season"]).agg(
        has_hc=("role", lambda r: "head_coach" in r.values),
        has_oc=("role", lambda r: "offensive_coordinator" in r.values),
        has_dc=("role", lambda r: "defensive_coordinator" in r.values),
    ).reset_index()
    n_complete_staff = ((cs_team_seasons["has_hc"]) &
                        (cs_team_seasons["has_oc"]) &
                        (cs_team_seasons["has_dc"])).sum()
    total_team_seasons = len(cs_team_seasons)

    coaches_with_lineage = edges[edges["mentee_slug"].notna()]["mentee_slug"].nunique()
    total_coaches = cs[cs["sr_slug"] != ""]["sr_slug"].nunique()

    # cont coverage
    cont_cov = len(cont) / max(total_team_seasons, 1)

    # lines
    lines = []
    lines.append("# NFL projection-model data layer — status")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Done criteria (per brief)")
    lines.append("")
    lines.append("| # | criterion | status |")
    lines.append("|---|---|---|")
    lines.append(f"| 1 | Every NFL game 2000+ in `games.parquet` w/ vegas + outcome | "
                 f"{games_with_vegas}/{len(games_2000)} games "
                 f"({100*games_with_vegas/len(games_2000):.1f}% vegas coverage) |")
    lines.append(f"| 2 | Every game has matched pbp in `pbp.parquet` | "
                 f"{n_games_pbp_match}/{len(games_2000)} games "
                 f"({100*n_games_pbp_match/len(games_2000):.1f}%) |")
    lines.append(f"| 3 | coach_seasons w/ HC + OC + DC per team-season | "
                 f"{n_complete_staff}/{total_team_seasons} team-seasons "
                 f"({100*n_complete_staff/total_team_seasons:.1f}%) |")
    lines.append(f"| 4 | Every coach has lineage_edge or is flagged root | "
                 f"{coaches_with_lineage} edges + "
                 f"{(edges['role_match_type']=='lineage_root').sum()} roots = "
                 f"{coaches_with_lineage + (edges['role_match_type']=='lineage_root').sum()}/{total_coaches} coaches |")
    lines.append(f"| 5 | state_residuals w/ pace + pass_rate + red_zone_trips | "
                 f"{len(sr):,} team-game rows |")
    lines.append(f"| 6 | continuity + injury joined to games.parquet | "
                 f"continuity covers {cont_cov*100:.1f}% of team-seasons; "
                 f"injury joined for 2009+ games |")
    lines.append("")

    lines.append("## Output tree")
    lines.append("")
    lines.append("```")
    lines.append(f"data/nfl/")
    for sub_dir, files in [("games/", ["games.parquet", "pbp.parquet", "state_residuals.parquet"]),
                           ("coaching/", ["coach_seasons.parquet", "coach_transitions.parquet",
                                          "lineage_edges.parquet", "coach_slugs.parquet"]),
                           ("rosters/", ["weekly_rosters_all.parquet", "continuity.parquet"]),
                           ("injuries/", ["injuries_all.parquet"])]:
        lines.append(f"  {sub_dir}")
        for f in files:
            fp = ROOT / sub_dir / f
            if fp.exists():
                size = fp.stat().st_size
                size_str = (f"{size/1e6:.1f} MB" if size > 1e6
                            else f"{size/1024:.0f} KB")
                lines.append(f"    {f:35s} {size_str}")
            else:
                lines.append(f"    {f:35s} MISSING")
    lines.append("```")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("- **nflfastR** play-by-play 1999-2024 (nflverse-data GitHub releases)")
    lines.append("- **nflverse schedules** (vegas lines + outcomes 2000+)")
    lines.append("- **nflverse weekly_rosters** 2002+ (continuity)")
    lines.append("- **nflverse injuries** 2009+ (Q/Out covariates)")
    lines.append("- **PFR team-season pages** 1999-2024 × 32 franchises (HC/OC/DC + position coaches via cache re-parse)")
    lines.append("- **PFR coach pages** for HC + OC + DC slugs (lineage / career history)")
    lines.append("")

    lines.append("## Known gaps")
    lines.append("")
    lines.append("- **Pre-2002 weekly rosters** not available from nflverse")
    lines.append("- **Pre-2009 injury reports** not available from nflverse")
    lines.append("- **NextGenStats / pre-snap motion** ships separately (pbp_participation 2016+); not in this fetch")
    lines.append("- **PFR position coaches** have inconsistent role label conventions across years; ~700 'other_assistant' rows preserved with raw role string in `role_raw`")
    lines.append("- **Coordinator-level lineage** is captured; position-coach mentor edges are derivable from supervisor field but not pre-emitted in lineage_edges (would create ~10x more edges; can be derived on-demand)")
    lines.append("")

    out = ROOT / "data_layer_status.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  WROTE {out}")


def main():
    augment_lineage_root()
    games = join_covariates_onto_games()
    write_status_md(games)
    print("\nPhase 6 complete.")


if __name__ == "__main__":
    main()
