"""Lay out the inputs where the analysis scripts read them, for one mode and one data window.

    python scripts/build/stage.py submitted 2024
    python scripts/build/stage.py corrected 2025

WINDOW: the study ran in two passes, and each test reproduces on the data it ran on:
  2024  the event studies, Layers 1 and 3, within-game, SVI, fingerprints, defensive chronic (seasons 2000-2024)
  2025  predictive validation, its robustness checks, propensity matching (through 2025, the LOSO refresh)
MODE:
  submitted  the panels exactly as the submitted paper used them (data/frozen)
  corrected  the panels rebuilt under one team-code standard (data/build/corrected, from panels.py corrected)
Everything is copied, never linked; a re-stage clears the previous layout (and any outputs left in it).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FROZEN = DATA / "frozen"
LAYOUT = ("games", "nfl", "coaching", "rosters", "raw", "nflfastR")


def season_of(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["season"], errors="coerce")
    if "game_id" in df.columns:   # a team-game the schedule never matched carries no season: read it off the id
        s = s.fillna(pd.to_numeric(df["game_id"].astype(str).str[:4], errors="coerce"))
    return s


def write(df: pd.DataFrame, *rels: str) -> None:
    for rel in rels:
        p = DATA / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)


def stage(mode: str, window: int) -> None:
    for d in LAYOUT:
        if (DATA / d).exists():
            shutil.rmtree(DATA / d)
    for f in DATA.glob("*.csv"):
        f.unlink()
    panels = FROZEN if mode == "submitted" else DATA / "build" / "corrected"
    sr = pd.read_parquet(FROZEN / "state_residuals.parquet" if mode == "submitted"
                         else panels / "games" / "state_residuals.parquet")
    games = pd.read_parquet(FROZEN / "games.parquet" if mode == "submitted" else panels / "games" / "games.parquet")
    cont = pd.read_parquet(FROZEN / "continuity.parquet" if mode == "submitted"
                           else panels / "rosters" / "continuity.parquet")
    pbp = pd.read_parquet(FROZEN / "pbp.parquet")
    pbp, sr = pbp[season_of(pbp) <= window], sr[season_of(sr) <= window]

    write(pbp, "games/pbp.parquet", "nfl/games/pbp.parquet")
    write(sr, "games/state_residuals.parquet", "nfl/games/state_residuals.parquet")
    write(games, "games/games.parquet", "nfl/games/games.parquet")
    write(cont, "rosters/continuity.parquet", "nfl/rosters/continuity.parquet")
    cs = pd.read_parquet(FROZEN / "coach_seasons.parquet")
    write(cs, "coaching/coach_seasons.parquet", "nfl/coaching/coach_seasons.parquet")
    write(pd.read_parquet(FROZEN / "draft_picks.parquet"), "nfl/combine/draft_picks.parquet")
    rost = pd.read_parquet(FROZEN / "rosters_2002_2024.parquet", columns=["season", "gsis_id", "position"])
    part = pd.read_parquet(FROZEN / "participation_2016_2024.parquet")
    dpbp = pd.read_parquet(FROZEN / "pbp_defense_2016_2024.parquet")
    for y in range(2016, 2025):
        write(rost[rost["season"] == y], f"nfl/rosters/weekly_rosters_{y}.parquet")
        write(part[part["season"] == y].drop(columns="season"), f"raw/participation/pbp_participation_{y}.parquet")
        write(dpbp[dpbp["season"] == y], f"nflfastR/play_by_play_{y}.parquet")
    (DATA / "raw" / "pfr").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FROZEN / "coaching_history.csv", DATA / "raw" / "pfr" / "coaching_history.csv")
    print(f"staged: {mode}, seasons through {window} ({len(sr):,} team-games, {len(pbp):,} plays)")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("submitted", "corrected") or sys.argv[2] not in ("2024", "2025"):
        sys.exit("usage: stage.py submitted|corrected 2024|2025")
    stage(sys.argv[1], int(sys.argv[2]))
