# Teammate Composition Signal Check — Pre-Registration

**Locked:** 2026-05-10, BEFORE any data analysis or panel construction.
**Status:** Pre-registered. No modifications to test players, cascade gates, decision rules, or thresholds permitted post-data-touch.

## Hypothesis (verbatim from spec)

**H_TC:** Per-snap target rate and yardage outcomes for skill-position players vary systematically by pre-snap teammate composition conditional on formation and personnel, beyond what season-mean player rates and opponent quality would predict.

**NULL:** Per-snap outcomes don't organize by teammate composition × formation cells; apparent variation is captured by player season-mean + opponent quality + game script alone.

## Test players (locked, no fourth)

| Slot | Player | gsis_id | Seasons | Treatment variation |
|---|---|---|---|---|
| TE | Travis Kelce | TBD on fetch | 2023, 2024 | Rashee Rice W1-4 injury 2024, Worthy 2024, MVS rotation, Pacheco out of backfield |
| WR2 | Garrett Wilson | TBD on fetch | 2024 | Adams trade arrival midseason → role shift Adams-absent → Adams-present → Adams-absent (Rodgers benched) |
| RB | Isiah Pacheco | TBD on fetch | 2023, 2024 | Kelce-on-field × Rice-on-field × formation interactions; ACL absence W4-W12 2024 |

## Data granularity (locked)

- **Primary source:** nflverse `pbp_participation_<year>.parquet` 2023-2025
- **Tags available:** offense_formation (SHOTGUN / SINGLEBACK / EMPTY / I_FORM / JUMBO / PISTOL / WILDCAT), offense_personnel (11 / 12 / 21 etc.), offense_players (list of gsis_ids), defenders_in_box, defense_personnel
- **NOT available without paid PFF/Sumer:** trips/bunch/3×1/2×2, slot-vs-outside, route-tree
- **Test will run at: formation × personnel × teammate-identity-set granularity.** Fine-grained alignment (PFF-style) is part of the granularity-check cascade but the fine level is data-blocked; report will surface this constraint per the spec.

## Cascade gates (5, locked)

A player's signal must clear ALL FIVE before claiming SIGNAL_DETECTED:

1. **Permutation null** — shuffle teammate-composition labels within the player's snaps, B=1000 reps. Observed η² must exceed the 95th percentile of the null distribution.
2. **Granularity comparison** — compute η² at three levels:
   - L1: personnel-only (11/12/21)
   - L2: formation + personnel + teammate-identity-set (primary test level)
   - L3: formation + personnel + teammate-identity-set + offense_formation_specific (finest level testable without PFF)
   Signal at L2 must match or exceed L1; signal at L3 may or may not exceed L2 (small-n risk).
3. **Replication** — signal must appear in ≥2 of 3 test players for HYPOTHESIS_PARTIAL, all 3 for HYPOTHESIS_CONFIRMED.
4. **Game-script confound control** — residualize per-snap outcome against (score margin, down, ydstogo, time remaining in half) via OLS before computing cell residual-mean and η².
5. **Opponent confound control** — residualize per-snap outcome against `opponent_strength_feature` (already-computed baseline feature, per-position trailing) before computing cell η².

## Decision rules (locked verbatim)

**Per player:**
- **SIGNAL_DETECTED**: η² > 0.10, p_emp < 0.05 after cascade, replicates on ≥1 other test player
- **WEAK_SIGNAL**: η² ∈ [0.05, 0.10], p_emp < 0.10, present in only 1 of 3
- **NULL**: η² < 0.05 across all test players (i.e., the test player itself is NULL)

**Cross-player:**
- **HYPOTHESIS_CONFIRMED**: all 3 test players show SIGNAL_DETECTED with consistent signal magnitudes
- **HYPOTHESIS_PARTIAL**: 2 of 3 show SIGNAL_DETECTED, pattern interpretable
- **HYPOTHESIS_REJECTED**: ≤1 of 3 shows SIGNAL_DETECTED, no systematic pattern

## Outcomes measured (per snap, locked)

For each snap the test player was on the field:

1. **`targeted`** (binary): did the QB throw at this player? (from receiver_player_id in pbp)
2. **`caught`** (binary, conditional on target): did the target convert to reception?
3. **`yards`** (continuous, conditional on target): yards gained on the play if targeted

Target rate is the primary outcome. Yards-per-target is the secondary outcome (smaller-n; will be tested but with looser interpretation). Catch rate is reported but not the primary signal.

## Universe

- Snaps where test player was on the field as an offensive player (offense_players list contains their gsis_id)
- Regular-season only (game_type == REG)
- Skill plays only (play_type in {pass, run, no_play}); excludes special teams, penalties that void the play
- For target-rate computation: pass plays only (play_type == pass)
- For yards-per-target: pass plays where receiver_player_id == test player's gsis_id

## Cell definition

Teammate composition cell = sorted tuple of (other skill-position gsis_ids in offense_players) × offense_formation × offense_personnel. Skill positions = QB/RB/WR/TE/FB. Cells with n < 10 snaps (per player) are pooled into an "other" bin to prevent overfit.

## Pre-locked thresholds (NOT adjustable post-data-touch)

| Threshold | Value |
|---|---:|
| η² for SIGNAL_DETECTED | > 0.10 |
| η² for WEAK_SIGNAL | [0.05, 0.10] |
| η² for NULL | < 0.05 |
| Permutation null p for SIGNAL_DETECTED | < 0.05 |
| Permutation null p for WEAK_SIGNAL | < 0.10 |
| Permutation reps | B = 1000 |
| Minimum cell n | 10 |
| Replication requirement | ≥2 of 3 for HYPOTHESIS_PARTIAL |

## Deliverables (locked, will be produced on completion)

1. `TEAMMATE_COMPOSITION_SIGNAL_CHECK.md` — disposition table at top, per-player + cross-player report, cascade results per player, granularity comparison discussion
2. `data/nfl/teammate_comp/per_player_signal.csv` — per-snap residual data for each test player
3. `data/nfl/teammate_comp/cell_breakdown.csv` — per (player × cell) rates and sample sizes
4. `data/nfl/teammate_comp/granularity_comparison.csv` — η² + p_emp at L1/L2/L3 per player

## Scope boundaries (locked)

- This is a **falsifiable signal check, not a build.** No forward projection infrastructure is constructed in this phase.
- If signal detected: justifies scoping a build for the next phase. Build itself is NOT in scope here.
- If signal not detected at formation+composition level but is at fine level: surface in report, recommend whether finer-level chase is worth committing PFF/Sumer data acquisition.
- If signal not detected at any level: confirms NULL hypothesis at testable granularity; build is not justified at currently-available data resolution.

## Pre-registration sign-off

This document is the contract. Test players, gates, thresholds, decision rules, and deliverables are locked. Modifications post-data-touch invalidate the test.

**Locked by:** Claude Code session, 2026-05-10, before any per-snap data was loaded.

---

*Next step: fetch pbp_participation 2023-2025, build per-snap panels, run cascade. No deviation from this contract.*
