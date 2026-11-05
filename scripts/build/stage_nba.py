"""Lay out the NBA inputs where scripts/nba read them (data/nba/inputs), for one mode.

    python scripts/build/stage_nba.py submitted
    python scripts/build/stage_nba.py corrected

MODE:
  submitted  the frozen NBA inputs exactly as the analysis used them (data/frozen/nba)
  corrected  the same files, with the two Basketball-Reference coaching panels put on the nba.com team codes the
             box scores and tracking use (NBA_CANON); in the submitted analysis Brooklyn, Charlotte and Phoenix
             never matched across that join
Everything is copied, never linked; a re-stage clears the previous layout.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen" / "nba"
OUT = REPO / "data" / "nba" / "inputs"
NBA_CANON = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}   # Basketball-Reference -> nba.com
BREF = ("coaching_change_flags.parquet", "coach_team_season.parquet")


def main(mode: str) -> None:
    if mode not in ("submitted", "corrected"):
        sys.exit("usage: stage_nba.py submitted|corrected")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    for f in sorted(FROZEN.glob("*.parquet")):
        if mode == "corrected" and f.name in BREF:
            d = pd.read_parquet(f)
            d["team_abbr"] = d["team_abbr"].replace(NBA_CANON)
            d.to_parquet(OUT / f.name, index=False)
        else:
            shutil.copyfile(f, OUT / f.name)
    print(f"staged NBA inputs ({mode}) in {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "submitted")
