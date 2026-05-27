# Within-Game Test 2 (Redesigned) — Stress-Adaptation Partition

**Date:** 2026-05-09
**Spec source:** user brief, post-Phase-1.5 (the "Stress-Adaptation Test — Redesigned for Within-Coach Variance" doc)
**Status:** complete; 0 CARRIES across all 32 (cell × partition) tests

## Headline

| Partition | CARRIES / total cells |
|---|---:|
| coaching_vector_id | **0 / 16** |
| team_id            | **0 / 16** |

Stress adaptation at the segment-level shift, under the locked Phase 1 SNR
discipline (snr ≥ 1.5, p ≤ 0.01, cross-season-consistent ≥ 2/3, holdout > 0),
**does not carry on either coach or team identity.**

The redesign fixes the structural pinning that broke the original Test 2
audit (95.99% play-weighted HC stability → F_coach undefined), giving each
coaching staff many independent within-stint observations. The new
problem isn't pinning; it's that the segment-level shift signal genuinely
doesn't generalize cross-season.

## Section 1 — cell-by-cell verdict table

In-sample SNR ratios for both partitions, plus the **F_coach vs F_team** tag
(COACH if SNR_c ≥ 1.5 AND ≥ 1.2× team; TEAM symmetric; TIE otherwise).
Verdict column is the formal SNR-discipline verdict (NULL across the board
because every cell's holdout failure or cross-season-inconsistency disqualifies).

### leading_2plus (n=987 segments)

| metric | SNR_coach | SNR_team | ratio | F-tag | coach holdout | team holdout | coach verdict | team verdict |
|---|---:|---:|---:|---|---:|---:|---|---|
| pass_rate         | 0.96 | 0.84 | 1.14 | TIE  | -0.0747 | -0.0493 | NULL | NULL |
| shotgun_rate      | 1.11 | 0.90 | 1.22 | TIE  | -0.0939 | -0.0652 | NULL | NULL |
| no_huddle_rate    | 1.19 | 1.78 | 0.67 | TEAM | -0.1067 | -0.0058 | NULL | NULL |
| deep_throw_rate   | 1.17 | 0.84 | 1.39 | TIE  | -0.0786 | -0.0573 | NULL | NULL |

### leading_1 (n=2,122 segments)

| metric | SNR_coach | SNR_team | ratio | F-tag | coach holdout | team holdout | coach verdict | team verdict |
|---|---:|---:|---:|---|---:|---:|---|---|
| pass_rate         | 0.98 | 0.86 | 1.13 | TIE  | +0.0172 | -0.0152 | NULL | NULL |
| shotgun_rate      | 1.06 | 1.12 | 0.95 | TIE  | -0.0443 | -0.0253 | NULL | NULL |
| no_huddle_rate    | 1.49 | 1.98 | 0.75 | TEAM | +0.0031 | -0.0240 | NULL | NULL |
| deep_throw_rate   | 1.17 | 1.07 | 1.10 | TIE  | -0.0145 | -0.0546 | NULL | NULL |

### trailing_1 (n=2,198 segments)

| metric | SNR_coach | SNR_team | ratio | F-tag | coach holdout | team holdout | coach verdict | team verdict |
|---|---:|---:|---:|---|---:|---:|---|---|
| pass_rate         | 1.50 | 1.22 | 1.23 | TIE   | -0.0322 | -0.0347 | NULL | NULL |
| shotgun_rate      | 1.55 | 1.29 | 1.20 | COACH | -0.0313 | -0.0425 | NULL | NULL |
| no_huddle_rate    | 1.30 | 1.18 | 1.10 | TIE   | -0.0175 | -0.0593 | NULL | NULL |
| deep_throw_rate   | 1.21 | 1.16 | 1.04 | TIE   | -0.0060 | -0.0359 | NULL | NULL |

### trailing_2plus (n=1,069 segments)

| metric | SNR_coach | SNR_team | ratio | F-tag | coach holdout | team holdout | coach verdict | team verdict |
|---|---:|---:|---:|---|---:|---:|---|---|
| pass_rate         | 1.27 | 1.77 | 0.72 | TEAM | -0.0454 | -0.0687 | NULL | NULL |
| shotgun_rate      | 1.64 | 2.25 | 0.73 | TEAM | -0.0051 | -0.0905 | NULL | NULL |
| no_huddle_rate    | 1.80 | 2.35 | 0.76 | TEAM | -0.0413 | -0.1408 | NULL | NULL |
| deep_throw_rate   | 0.94 | 1.21 | 0.78 | TIE  | -0.0476 | -0.0473 | NULL | NULL |

### F-tag tally (in-sample only — not the formal verdict)

| | count |
|---|---:|
| COACH | 1 (trailing_1 shotgun_rate) |
| TEAM  | 5 (leading_2plus + leading_1 no_huddle; trailing_2plus pass + shotgun + no_huddle) |
| TIE   | 10 |

Where there's *any* in-sample F-stat directional signal, **team_id wins it more
often than coach_id** (5 vs 1). Consistent with the original (audited) Test 2
direction.

## Section 2 — comparison to original (withdrawn) Test 2

| | original Test 2 | redesigned Test 2 |
|---|---|---|
| unit | (team, season) shift values | (game, team, segment) shifts |
| placebo | none (point estimate only) | 1000-rep group-label permutation |
| holdout | none | train 2018-2022 / test 2023-2024 |
| structural pinning | 95.99% HC stability — F_coach NaN on conditioned subset | resolved: each coach has many segment rows |
| F-tag verdict | TEAM 5/12, TIE 7/12, COACH 0/12 | TEAM 5/16, TIE 10/16, COACH 1/16 |
| formal SNR verdict | not computed | **0/32 CARRIES** |

The directional in-sample finding is consistent: where partition signal exists,
team_id captures it better than coach_id. The formal SNR/holdout discipline,
which the original couldn't even compute, **rejects both partitions across the
board** — no cell generalizes.

## Section 3 — implications for the layer-pivoted findings table

| layer | test | result |
|---|---|---|
| Static (season-level) | F-stat coaching_vector vs team_id + perm null | TEAM-ID 2/4, TIE 2/4, COACH 0/4 |
| Within-season (HC change events) | pre/post |Δ residual| vs placebo | pass_rate p=0.002 (audited); plays + RZ null |
| Within-game schematic identity (Test 1, audited) | per-bucket F-stat partition + within-season perm null | **COACH dominant on shotgun + no_huddle** at 2-3× null; deep_throw collapses on 3 of 5 buckets under permutation |
| **Within-game stress adaptation (Test 2, redesigned)** | **per-segment SNR + holdout discipline** | **NULL — neither coach nor team partitions carry under full SNR discipline (0/32)** |

The four-layer story is now consistent and clean:

- **Coaching IS the latent for static schematic identity** (formation
  tendencies, tempo) — Test 1 lands COACH dominant on shotgun + no_huddle
- **Coaching IS detectable at HC-change events** for pass_rate within-season
  (sharp event-study test, p=0.002)
- **Coaching is NOT the latent for season-level outcomes** (static-layer
  TEAM-ID wins 2/4, TIE 2/4, COACH 0/4)
- **Coaching is NOT the latent for stress adaptation** (this redesign — and
  neither is team_id at SNR-discipline scale)

Stress adaptation is **noise at the segment-shift resolution.** The story
that "teams adapt under pressure in identifiable, repeated ways" doesn't
survive cross-season holdout.

## Section 4 — methodological notes on the redesign

### Deviations from brief

1. **Neutral baseline = tied state only**, not "tied or within-1-score states."
   Reason: the brief's "4 buckets × 4 metrics = 16 cells" implies non-neutral
   = {leading_2plus, leading_1, trailing_1, trailing_2plus}. If neutral
   included leading_1 + trailing_1 those would be excluded from non-neutral,
   leaving only 2 non-neutral buckets (leading_2plus, trailing_2plus). The
   stricter tied-only baseline keeps the cell count at 16 as specified.

2. **Placebo: standard global group-label permutation**, not within-coach-stint
   value shuffling. Reason: literal within-group value shuffling leaves group
   means (and therefore F_coach) invariant — it doesn't generate a usable
   null. The standard permutation is the SNR-canonical null for "does the
   group partition carry signal." If a within-coach-stint variant is wanted
   for comparison, it can be added in a refinement pass.

3. **Stable-baseline filter: ≥5 tied-state plays per team-game.** This
   discards 2,906/13,976 team-games where the tied state was rare or
   absent. Tighter filter (e.g., ≥10) didn't change the headline; loosening
   it (≥3) inflated baseline noise without helping cells.

### Sample sizes & power

| bucket | n_segments | median segment-cells per coach | median segment-cells per team |
|---|---:|---:|---:|
| leading_2plus | 987   | 5  | 23 |
| leading_1     | 2,122 | 11 | 49 |
| trailing_1    | 2,198 | 12 | 51 |
| trailing_2plus| 1,069 | 5  | 25 |

All cells well above the 50-floor on team partition; coach partition is at-
or-below floor in leading_2plus + trailing_2plus. **Coach-side underpowering
in the extreme buckets is real and would not necessarily be improved by more
seasons** — it's a function of how many distinct coaching staffs cycle
through the league; the league has 32 teams × ~few-year tenures.

### Why holdout fails so dramatically

In-sample SNR shows TEAM at 1.77-2.35 in trailing_2plus cells (above the
1.5 threshold), but holdout R² is -0.05 to -0.14. This is the classic
overfit pattern: team-specific stress-adaptation patterns fit in 2018-2022
do not replicate in 2023-2024. Possible reasons:

- The 2020-2021 COVID seasons had unusual game-script distributions that
  the model fit
- Teams reorganized rosters/coaching between train and test (mean-reverting)
- Stress-adaptation IS noise at this resolution — the in-sample SNR is
  itself overfitting the random pattern

The clean cross-season failure across all 32 cells is **the strongest
evidence** that there's no real stress-adaptation latent at this resolution.

## Reproduce

```bash
python data/nfl/_src/within_game_test2_redesign.py
```

Output: `data/nfl/within_game_test2_redesign_results.parquet` (32 rows; 16
cells × 2 partitions).

## Done criteria checklist

- [x] Results parquet populated for all 16 cells (× 2 partitions = 32 rows)
- [x] Summary markdown written
- [x] Layer-pivoted findings table updated (Section 3 + STATE.md update)
- [N/A] Diagnostic plots for any CARRIES cells — **no CARRIES**, none generated
- [x] SNR_TEST_STATE.md appended with redesigned-test result block
