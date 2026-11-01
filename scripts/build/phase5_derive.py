"""Phase 5: derive computed artifacts.

Inputs:
  data/nflfastR/play_by_play_{1999..2024}.parquet
  data/nfl/games/games.parquet (vegas + outcome)
  data/nfl/coaching/coach_seasons.parquet
  data/nfl/rosters/weekly_rosters_all.parquet
  data/pfr/raw_coaches/{slug}                (PFR coach pages)

Outputs:
  data/nfl/games/pbp.parquet                    (concat 1999-2024)
  data/nfl/games/state_residuals.parquet        (expected vs realized per game)
  data/nfl/coaching/lineage_edges.parquet       (mentor-mentee per season)
  data/nfl/coaching/coach_transitions.parquet   (parsed from PFR coach pages)
  data/nfl/rosters/continuity.parquet           (team-season returning-snap fraction)
"""
from __future__ import annotations
import re
import time
from pathlib import Path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

NFLFASTR_DIR = Path(__file__).resolve().parents[2] / "data" / "nflfastR"
COACHING_DIR = Path(__file__).resolve().parents[2] / "data" / "nfl" / "coaching"
GAMES_DIR = Path(__file__).resolve().parents[2] / "data" / "nfl" / "games"
ROSTERS_DIR = Path(__file__).resolve().parents[2] / "data" / "nfl" / "rosters"
COACH_PAGES = Path(__file__).resolve().parents[2] / "data" / "raw" / "pfr" / "raw_coaches"


# ---------- 1. CONCAT PBP ----------
def concat_pbp():
    print("\n=== concat pbp 1999-2024 ===")
    out = GAMES_DIR / "pbp.parquet"
    if out.exists() and out.stat().st_size > 100_000_000:
        print(f"  already exists ({out.stat().st_size/1e6:.0f} MB), skipping")
        return
    files = sorted(NFLFASTR_DIR.glob("play_by_play_*.parquet"))
    print(f"  concatenating {len(files)} per-season parquets")
    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        dfs.append(df)
        print(f"    {fp.name}: {len(df):,} rows")
    full = pd.concat(dfs, ignore_index=True)
    print(f"  total rows: {len(full):,}")
    # cast object cols to string for parquet
    for c in full.columns:
        if full[c].dtype == "object":
            full[c] = full[c].astype("string")
    full.to_parquet(out, index=False)
    print(f"  WROTE {out} ({out.stat().st_size/1e6:.1f} MB)")


# ---------- 2. STATE RESIDUALS ----------
def state_residuals():
    print("\n=== state_residuals (expected vs realized per game) ===")
    games = pd.read_parquet(GAMES_DIR / "games.parquet")
    pbp = pd.read_parquet(GAMES_DIR / "pbp.parquet")

    # per-team-per-game realized stats from pbp
    # qualified plays: regular pass/rush, exclude kneels/spikes/2pt
    play = pbp[(pbp["play_type"].isin(["pass", "run"]))
               & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)
               & (pbp["two_point_attempt"] != 1)
               & pbp["posteam"].notna()].copy()
    play["is_pass"] = (play["play_type"] == "pass").astype(int)
    play["is_red_zone"] = (play["yardline_100"] <= 20).astype(int)

    realized = (play.groupby(["game_id", "posteam"]).agg(
        plays=("play_id", "count"),
        pass_rate=("is_pass", "mean"),
        red_zone_plays=("is_red_zone", "sum"),
        epa_per_play=("epa", "mean"),
    ).reset_index())

    # Pace = plays per game (proxy)
    # Red zone TRIPS = unique drives that hit red zone (use yardline_100<=20)
    rz = (play[play["is_red_zone"] == 1]
          .groupby(["game_id", "posteam", "drive"])["play_id"].count()
          .reset_index().groupby(["game_id", "posteam"]).size()
          .reset_index(name="red_zone_trips"))
    realized = realized.merge(rz, on=["game_id", "posteam"], how="left")
    realized["red_zone_trips"] = realized["red_zone_trips"].fillna(0).astype(int)

    # Compute league averages by season -- pbp already has season
    season_avgs = (play.groupby("season").agg(
        league_pass_rate=("is_pass", "mean"),
        league_plays_per_pteam_game=("play_id", lambda s: len(s) / (32 * 16)),
    ).reset_index())

    # join games + spread to derive expected pass rate
    games_keep = games[["game_id", "season", "week", "home_team", "away_team",
                        "home_score", "away_score", "spread_line", "total_line",
                        "result"]].copy()

    # Each game has TWO posteam rows; determine each team's spread perspective
    home_view = games_keep.copy()
    home_view["posteam"] = home_view["home_team"]
    home_view["team_spread"] = -home_view["spread_line"]  # if spread_line is favorite-perspective
    home_view["team_score"] = home_view["home_score"]
    home_view["opp_score"] = home_view["away_score"]
    away_view = games_keep.copy()
    away_view["posteam"] = away_view["away_team"]
    away_view["team_spread"] = away_view["spread_line"]
    away_view["team_score"] = away_view["away_score"]
    away_view["opp_score"] = away_view["home_score"]
    persp = pd.concat([home_view, away_view], ignore_index=True)

    full = realized.merge(persp[["game_id", "posteam", "season", "week",
                                  "team_spread", "total_line",
                                  "team_score", "opp_score"]],
                          on=["game_id", "posteam"], how="left")
    full = full.merge(season_avgs, on="season", how="left")

    # Expected pass rate: trailing teams pass more.
    # Crude model: pass_rate_exp = league + 0.012 * (-team_spread)
    # i.e. underdogs (positive spread) pass more
    full["expected_pass_rate"] = full["league_pass_rate"] + 0.012 * full["team_spread"]
    full["pass_rate_residual"] = full["pass_rate"] - full["expected_pass_rate"]

    # Expected pace = league plays/game baseline + game-total-side adjustment
    # Crude model: pace_exp = league + 0.4 * (total_line - league_total_avg) / league_total_std
    # League totals
    season_total_avg = games.groupby("season")["total_line"].mean().reset_index().rename(
        columns={"total_line": "season_total_avg"})
    full = full.merge(season_total_avg, on="season", how="left")
    full["expected_plays"] = full["league_plays_per_pteam_game"] + \
                             0.4 * (full["total_line"] - full["season_total_avg"])
    full["plays_residual"] = full["plays"] - full["expected_plays"]

    # Expected red zone trips: 0.18 * total_line / 7 baseline (each TD = ~7 pts, RZ trip ~ 0.7 TD)
    # Crude: expected_rz = (total_line + team_spread*-0.5) / 14
    full["expected_red_zone_trips"] = (full["total_line"]
                                        - 0.5 * full["team_spread"]) / 14.0
    full["red_zone_trips_residual"] = (full["red_zone_trips"]
                                        - full["expected_red_zone_trips"])

    out = GAMES_DIR / "state_residuals.parquet"
    full.to_parquet(out, index=False)
    print(f"  WROTE {out} ({len(full):,} team-game rows)")
    print(f"  pass_rate_residual mean={full['pass_rate_residual'].mean():.4f} "
          f"std={full['pass_rate_residual'].std():.4f}")
    print(f"  plays_residual mean={full['plays_residual'].mean():.4f} "
          f"std={full['plays_residual'].std():.4f}")
    print(f"  red_zone_trips_residual mean={full['red_zone_trips_residual'].mean():.4f} "
          f"std={full['red_zone_trips_residual'].std():.4f}")


# ---------- 3. LINEAGE EDGES ----------
def lineage_edges():
    print("\n=== lineage_edges (mentor-mentee from supervisor field) ===")
    cs = pd.read_parquet(COACHING_DIR / "coach_seasons.parquet")
    # For each coach-season-team-role with a supervisor, find who held that supervisor role on
    # the same team-season -> emit (mentor, mentee, season, role)
    edges = []
    cs_indexed = cs.set_index(["team", "season", "role"])
    for r in cs.itertuples(index=False):
        if not r.supervisor:
            continue
        # locate supervisor in same team-season holding that role
        try:
            sup_rows = cs_indexed.loc[(r.team, r.season, r.supervisor)]
            if isinstance(sup_rows, pd.Series):
                sup_rows = sup_rows.to_frame().T
        except KeyError:
            continue
        for sup in sup_rows.itertuples(index=False):
            if sup.coach_name == r.coach_name:
                continue
            edges.append({
                "mentor_name": sup.coach_name, "mentor_slug": sup.sr_slug,
                "mentor_role": r.supervisor,
                "mentee_name": r.coach_name, "mentee_slug": r.sr_slug,
                "mentee_role": r.role,
                "team": r.team, "season": r.season,
                "role_match_type": "direct_report",
            })
    edge_df = pd.DataFrame(edges)
    out = COACHING_DIR / "lineage_edges.parquet"
    edge_df.to_parquet(out, index=False)
    print(f"  WROTE {out} ({len(edge_df):,} mentor-mentee edges)")

    # Coverage check: how many unique mentees have at least one edge
    mentees_with_edge = edge_df["mentee_slug"].dropna().unique()
    all_coaches = cs["sr_slug"][cs["sr_slug"] != ""].unique()
    coverage = len(set(mentees_with_edge) & set(all_coaches)) / max(len(all_coaches), 1)
    print(f"  coach lineage coverage: "
          f"{len(set(mentees_with_edge) & set(all_coaches))}/{len(all_coaches)} "
          f"({100*coverage:.1f}%)")


# ---------- 4. COACH TRANSITIONS ----------
def coach_transitions():
    print("\n=== coach_transitions (parse PFR coach pages) ===")
    files = sorted(COACH_PAGES.glob("*.htm"))
    if not files:
        print("  no cached coach pages found, skipping")
        return
    rows = []
    fail = 0
    for fp in files:
        slug = fp.name
        try:
            html = fp.read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "html.parser")
            # career table id can vary; try common ones
            t = soup.find("table", id="all_coaching_history") \
                or soup.find("table", id="coaching_history") \
                or soup.find("table", id="all_history") \
                or soup.find("table", attrs={"id": re.compile("history")})
            if t is None:
                # try all tables and pick first
                tabs = soup.find_all("table")
                t = tabs[0] if tabs else None
            if t is None:
                fail += 1
                continue
            # name from h1
            h1 = soup.find("h1")
            name = h1.get_text(strip=True) if h1 else slug
            tbody = t.find("tbody")
            if not tbody:
                fail += 1
                continue
            for tr in tbody.find_all("tr"):
                row = {}
                for cell in tr.find_all(["th", "td"]):
                    ds = cell.get("data-stat")
                    if ds:
                        row[ds] = cell.get_text(" ", strip=True)
                if not row:
                    continue
                rows.append({
                    "coach_slug": slug, "coach_name": name,
                    "year": row.get("year_id") or row.get("year") or row.get("season") or "",
                    "team_id": row.get("team_id") or row.get("team") or "",
                    "league_id": row.get("league_id") or row.get("lg_id") or "",
                    "role": row.get("title") or row.get("position") or row.get("role") or "",
                    "raw_row": str(row),
                })
        except Exception as e:
            fail += 1
            continue
    df = pd.DataFrame(rows)
    print(f"  parsed {len(files) - fail}/{len(files)} pages, {len(df):,} career rows")
    out = COACHING_DIR / "coach_transitions.parquet"
    if len(df):
        df.to_parquet(out, index=False)
        print(f"  WROTE {out}")


# ---------- 5. ROSTER CONTINUITY ----------
def roster_continuity():
    print("\n=== roster continuity (returning-snap fraction per team-season) ===")
    rosters = pd.read_parquet(ROSTERS_DIR / "weekly_rosters_all.parquet")
    # Need a stable player id; use 'gsis_id' if present, else 'pfr_player_id', else 'player_id'
    id_col = next((c for c in ["gsis_id", "pfr_player_id", "player_id", "esb_id"]
                   if c in rosters.columns), None)
    if id_col is None:
        print("  no player id column found; skipping continuity")
        return
    # 'season' col may or may not be in rosters; we have _season we set during fetch
    season_col = "season" if "season" in rosters.columns else "_season"
    # Roster per (team, season): unique players who appeared
    roster_set = (rosters.dropna(subset=[id_col])
                          .groupby([season_col, "team"])[id_col]
                          .agg(set).reset_index().rename(columns={season_col: "season"}))
    roster_set = roster_set.sort_values(["team", "season"])
    cont_rows = []
    for team, grp in roster_set.groupby("team"):
        grp = grp.sort_values("season").reset_index(drop=True)
        for i in range(1, len(grp)):
            prior = grp.iloc[i - 1]
            cur = grp.iloc[i]
            shared = prior[id_col] & cur[id_col]
            roster_continuity_pct = len(shared) / max(len(prior[id_col]), 1)
            cont_rows.append({
                "team": team, "season": cur["season"],
                "n_players": len(cur[id_col]),
                "n_returning": len(shared),
                "roster_continuity_pct": roster_continuity_pct,
            })
    df = pd.DataFrame(cont_rows)
    out = ROSTERS_DIR / "continuity.parquet"
    df.to_parquet(out, index=False)
    print(f"  WROTE {out} ({len(df):,} team-season rows)")
    print(f"  mean continuity: {df['roster_continuity_pct'].mean():.3f}")


def main():
    t0 = time.time()
    concat_pbp()
    state_residuals()
    lineage_edges()
    coach_transitions()
    roster_continuity()
    print(f"\nTotal Phase 5 wall-clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
