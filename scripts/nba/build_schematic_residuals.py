"""Build per-team-game schematic residuals: pace, 3PT rate, AST%.

NBA Projections sibling data borrowing:
  D:/NBA Projections/data/parquet/historical_box_scores.parquet
    (player-game box scores 2019-20 through 2025-26)
  D:/NBA Projections/data/parquet/coach_team_season.parquet
    (HC per team-season 2014-15 through 2024-25)

Outputs (local, gitignored):
  D:/NBA Projections/coaching_signal/data/parquet/team_game_residuals.parquet

Schematic metrics (NFL pass_rate / rz_shotgun analog for NBA):
  pace_residual:        possessions/game minus season mean (HC tempo identity)
  three_pt_rate_residual: 3PA/FGA minus season mean (HC shot selection identity)
  ast_rate_residual:    AST/FGM minus season mean (ball-movement philosophy)

Possessions formula: FGA - OREB + TO + 0.44*FTA (per-team-game).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "nba" / "inputs"   # staged per mode: scripts/build/stage_nba.py
OUT = Path(__file__).resolve().parents[2] / "data" / "nba" / "data" / "parquet"
OUT.mkdir(parents=True, exist_ok=True)


def build_team_game_aggregate() -> pd.DataFrame:
    """Aggregate player box scores -> team-game totals."""
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()

    tg = (bs.groupby(["game_id", "season", "team_abbr", "game_date", "is_home"],
                       as_index=False)
            .agg(FGM=("FGM", "sum"), FGA=("FGA", "sum"),
                 FG3M=("FG3M", "sum"), FG3A=("FG3A", "sum"),
                 FTM=("FTM", "sum"), FTA=("FTA", "sum"),
                 OREB=("OREB", "sum"), DREB=("DREB", "sum"),
                 AST=("AST", "sum"), TOV=("TOV", "sum"),
                 PTS=("PTS", "sum"), STL=("STL", "sum"), BLK=("BLK", "sum")))
    tg["possessions"] = tg["FGA"] - tg["OREB"] + tg["TOV"] + 0.44 * tg["FTA"]
    tg["three_pt_rate"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    tg["ast_rate"] = tg["AST"] / tg["FGM"].clip(lower=1)
    # Game-level week proxy: sort by game_date within season, rank
    tg = tg.sort_values(["season", "team_abbr", "game_date"])
    tg["game_in_season"] = tg.groupby(["season", "team_abbr"]).cumcount() + 1
    return tg


def add_residuals(tg: pd.DataFrame) -> pd.DataFrame:
    """Per metric: subtract season-league-mean."""
    for met in ["possessions", "three_pt_rate", "ast_rate"]:
        season_mean = tg.groupby("season")[met].transform("mean")
        tg[f"{met}_residual"] = tg[met] - season_mean
    return tg


def main():
    print("=== BUILD NBA SCHEMATIC RESIDUALS ===\n")
    tg = build_team_game_aggregate()
    print(f"team-game rows: {len(tg):,}, seasons: "
          f"{sorted(tg['season'].unique())}")
    tg = add_residuals(tg)
    print("\nresidual sanity:")
    for met in ["possessions_residual", "three_pt_rate_residual", "ast_rate_residual"]:
        print(f"  {met:<30} mean={tg[met].mean():+.4f} std={tg[met].std():.4f}")
    out = OUT / "team_game_residuals.parquet"
    tg.to_parquet(out, index=False)
    print(f"\nWROTE {out} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
