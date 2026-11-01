"""Build the three analysis panels from data/frozen with the ORIGINAL builders.

    python scripts/build/panels.py submitted   # rebuild the panels the paper used, then prove they equal data/frozen
    python scripts/build/panels.py corrected   # the same builders after one team-code standard (see README, "Corrected")

Panels:
  state_residuals.parquet  phase5_derive.state_residuals()        expected vs realized per team-game
  continuity.parquet       phase5_derive.roster_continuity()      returning-roster share per team-season
  games.parquet            phase6_assemble.join_covariates_onto_games()   schedule + continuity + injury covariates
The builders are the research code verbatim except for their path constants (scripts/build/phase5_derive.py,
phase6_assemble.py). Output: data/build/<mode>/.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_derive as p5  # noqa: E402
import phase6_assemble as p6  # noqa: E402

# One team-code standard: nflverse's current codes. The play-by-play already uses them; the schedule, the
# rosters and the injury reports use the codes of their time (STL, SD, OAK) and the rosters add their own
# (SL, ARZ, BLT, CLV, HST). Joining across those silently drops the team-games that do not match.
CANON = {"STL": "LA", "SL": "LA", "LAR": "LA", "SD": "LAC", "OAK": "LV",
         "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}


def build(mode: str) -> Path:
    out = REPO / "data" / "build" / mode
    for d in ("games", "rosters", "injuries"):
        (out / d).mkdir(parents=True, exist_ok=True)
    schedule = pd.read_parquet(FROZEN / "schedule.parquet")
    rosters = pd.read_parquet(FROZEN / "rosters_2002_2024.parquet", columns=["season", "team", "gsis_id"])
    injuries = pd.read_parquet(FROZEN / "injuries_2009_2025.parquet")
    if mode == "corrected":
        for c in ("home_team", "away_team"):
            schedule[c] = schedule[c].replace(CANON)
        rosters["team"] = rosters["team"].astype(str).replace(CANON)
        injuries["team"] = injuries["team"].astype(str).replace(CANON)
    shutil.copyfile(FROZEN / "pbp.parquet", out / "games" / "pbp.parquet")
    schedule.to_parquet(out / "games" / "games.parquet", index=False)
    rosters.to_parquet(out / "rosters" / "weekly_rosters_all.parquet", index=False)
    injuries.to_parquet(out / "injuries" / "injuries_all.parquet", index=False)

    p5.GAMES_DIR, p5.ROSTERS_DIR = out / "games", out / "rosters"
    p5.state_residuals()
    p5.roster_continuity()
    p6.GAMES, p6.ROSTERS, p6.INJURIES = out / "games", out / "rosters", out / "injuries"
    p6.join_covariates_onto_games()
    return out


def verify(out: Path) -> bool:
    """The rebuilt panels must equal the frozen ones exactly (values; dtypes may differ by writer)."""
    ok = True
    checks = [("games/state_residuals.parquet", "state_residuals.parquet", ["game_id", "posteam"]),
              ("rosters/continuity.parquet", "continuity.parquet", ["team", "season"]),
              ("games/games.parquet", "games.parquet", ["game_id"])]
    for built, frozen, key in checks:
        f = pd.read_parquet(FROZEN / frozen)
        b = pd.read_parquet(out / built)
        b = b[b["season"] <= f["season"].max()] if "game_id" in key and built.endswith("games.parquet") else b
        b = b[list(f.columns)]
        try:
            pd.testing.assert_frame_equal(b.sort_values(key).reset_index(drop=True),
                                          f.sort_values(key).reset_index(drop=True), check_dtype=False)
            print(f"  OK    {frozen}: rebuilt == frozen ({len(f):,} rows)")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {frozen}: {str(e).splitlines()[0]}")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "submitted"
    if mode not in ("submitted", "corrected"):
        sys.exit("usage: panels.py submitted|corrected")
    out = build(mode)
    if mode == "submitted":
        sys.exit(0 if verify(out) else 1)
