# Frozen inputs: what they are and where they come from

These files are the inputs the paper's results were computed from, cut to the columns the code reads.
`MANIFEST.sha256` holds the SHA-256 of each file. `reproduce.sh` refuses to run if any file differs from it.

They are frozen on purpose. Rebuilding from the live sources would not reproduce the paper:
- nflverse revises past seasons.
- The coaching history was collected from Pro Football Reference pages that can no longer be fetched by a plain
  script.

`bash reproduce.sh verify` rebuilds the three analysis panels (`state_residuals`, `continuity`, `games`) from the
raw files here, using the original builders, and checks that they equal the frozen panels.

| File | Contents | Source |
|---|---|---|
| `pbp.parquet` | play-by-play 1999–2025, 24 columns | nflverse `pbp` releases |
| `pbp_defense_2000_2024.parquet` | play-by-play columns the defensive decomposition and its 2000-2024 extension read | nflverse `pbp` releases |
| `participation_2016_2024.parquet` | defensive players on the field per play | nflverse `pbp_participation` releases |
| `rosters_2002_2024.parquet` | season, team, player id, position | nflverse `weekly_rosters` releases |
| `draft_picks.parquet` | player id, pick number | nflverse `draft_picks` release |
| `injuries_2009_2025.parquet` | season, week, team, report status | nflverse `injuries` releases |
| `schedule.parquet` | game id, teams, scores, point spread and total | nflverse `schedules` release |
| `games.parquet` | the schedule plus continuity and injury covariates | built: `phase6_assemble.join_covariates_onto_games` |
| `state_residuals.parquet` | expected vs realized pass rate, plays and red-zone trips per team-game | built: `phase5_derive.state_residuals` |
| `continuity.parquet` | returning-roster share per team-season | built: `phase5_derive.roster_continuity` |
| `coach_seasons.parquet` | coach, team, season, role, record | compiled from Pro Football Reference team pages |
| `coaching_history.csv` | head coach and coordinators per team-season, mid-season splits | compiled from Pro Football Reference team pages |
| `nba/historical_box_scores.parquet` | NBA player box scores, 2014-15 onward | nba.com, through `nba_api` |
| `nba/player_tracking_game.parquet` | NBA player tracking per game | nba.com, through `nba_api` |
| `nba/coach_team_season.parquet` | NBA head coach, record and mid-season changes per team-season | compiled from Basketball-Reference team pages |
| `nba/coaching_change_flags.parquet` | NBA coaching-change flags per team-season | derived from `coach_team_season` |

## Attribution

**nflverse.** Play-by-play, participation, rosters, draft picks, injuries and schedules are from the nflverse
project (https://github.com/nflverse/nflverse-data), released under CC-BY-4.0.

**Pro Football Reference.** The coaching files hold historical facts: who coached which team, in what role,
in which season, with what record. They were compiled by our own parser from Pro Football Reference team pages
(https://www.pro-football-reference.com), which we cite as the source. Each row carries its `source_url`. No
page content, page layout or Sports Reference metric is included. `sr_slug` is Sports Reference's own coach
identifier, kept so a reader can check any row against its page.

**nba.com.** The NBA box scores and tracking data are public nba.com statistics, collected through the
open-source `nba_api` client (https://github.com/swar/nba_api) and cited here as their source.

**Basketball-Reference.** The NBA coaching files hold historical facts: who coached which team, in which season,
with what record. They were compiled from Basketball-Reference team pages (https://www.basketball-reference.com),
which we cite as the source. Each row carries its `source_url`.

This is a non-commercial, open-source replication package. Every dataset is included exactly as the analysis
used it, so that the results can be reproduced, and each is attributed to its source above.

No betting odds or moneylines are included. The point spread and total are used only to set each team-game's
expected pass rate, plays and red-zone trips.
