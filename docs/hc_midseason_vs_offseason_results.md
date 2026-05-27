# HC Mid-Season vs Offseason Separation — Cleaner Natural Experiment

**Date:** 2026-05-15
**Status:** Paper-grade finding. Refines Layer 2a's headline `HC change → pass_rate shift 1.36x p=0.002` by separating two distinct natural experiments. The pass-rate effect lives **entirely in mid-season** events. Offseason HC changes are pass-rate null but show a previously-unisolated **play volume** signal.

## Headline

| Subset | Metric | n_real | n_plac | \|d\|_real | \|d\|_plac | ratio | p_perm | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Mid-season** | pass_rate_residual | 45 | 700 | 0.072 | 0.053 | **1.36** | **0.002** | **STAFF SHIFT MOVES NEEDLE** |
| Mid-season | plays_residual | 45 | 700 | 3.93 | 3.56 | 1.10 | 0.187 | null |
| Mid-season | rz_trips_residual | 45 | 700 | 0.61 | 0.57 | 1.07 | 0.256 | null |
| Offseason | pass_rate_residual | 142 | 347 | 0.062 | 0.061 | 1.02 | 0.428 | **null** |
| **Offseason** | plays_residual | 142 | 347 | 4.59 | 3.93 | **1.17** | **0.021** | **STAFF SHIFT MOVES NEEDLE** |
| Offseason | rz_trips_residual | 142 | 347 | 0.60 | 0.59 | 1.02 | 0.416 | null |

Mid-season placebo pool is the within-season fake-split design from the
original Layer 2a (n=700 (team-season, fake_split) pairs). Offseason placebo
is HC AND OC continuity team-seasons (n=347).

## The key refinement

Layer 2a's reported 1.36x pass_rate effect was conflating two mechanisms.
Once separated:

- **Mid-season HC firings (n=49 events)** drive the pass_rate signal.
  Interim coaches systematically shift the play-call mix — likely a
  combination of "play to remaining strengths" / "shake things up" /
  install-known-plays-fast — and the residual pass-rate moves cleanly.
- **Offseason HC changes (n=164 events)** do NOT shift the average
  team-level pass-rate. New HCs' schematic identity gets blended with
  the roster and coordinator they inherit; the magnitude of the average
  pre/post shift is indistinguishable from continuous-staff placebo.

This complicates and improves the headline finding. It does NOT contradict
the predictive-validation result that retread HC profiles forecast pass-rate
shift direction (r=+0.386, p=0.004, slope=0.30). What it says is:

- The **direction** of the shift is forecastable from the incoming HC's
  prior profile (retread predictive validation, n=41).
- The **average magnitude** of shifts across all offseason HC changes is
  zero — different HCs go in different directions and they cancel out at
  population level.

These two facts are consistent. Forecastability lives in the SIGN
(predicted by coach-prior profile diff), not in the population-level mean
absolute shift.

## The new finding — offseason HC affects play volume

The offseason `plays_residual` cell (ratio 1.17, p=0.021) is a previously-
unisolated signal. Offseason HC changes move per-game play volume by an
absolute mean of 4.59 plays/game, 17% more than the 3.93 plays/game of
continuous-staff placebo. This is structurally consistent: a new HC often
arrives with a pace philosophy (no-huddle ratio, in-game tempo,
clock-management style) that affects how many snaps per game the team
runs, even when the pass-vs-run split stays similar.

This signal didn't surface in Layer 2a because it was diluted by the
mid-season events (where interims rarely change pace). Separating the
two natural experiments cleans up both: mid-season owns pass-rate;
offseason owns plays/game.

## Mechanistic synthesis

| Event | Owns | Why |
|---|---|---|
| Mid-season HC change | Pass-rate identity shift | Interim shake-up; play to roster's remaining strengths |
| Offseason HC change | Pace / play volume shift | New HC's tempo philosophy; pre-game prep on tempo |
| Offseason HC change × prior coach profile | Pass-rate **direction** shift | Schematic-identity portability (Layer 5a) |
| Offseason OC change | RZ schematic fine-grain | OC owns red-zone playbook; portable (Layer 5b) |

The full coordinator-role asymmetry framework now reads:

- HC: macro philosophy at TWO temporal resolutions — mid-season firings
  shift pass-rate (crisis-driven play-mix change), offseason hires shift
  pace (deliberate tempo philosophy)
- OC: red-zone schematic identity (deliberate offseason hires)
- DC: structural null at all resolutions tested (personnel-driven, not
  scheme-driven)

## Implications

### Paper

This separation IS the natural-experiment cleanup the methodology brief
asked for. Three improvements to the Sloan submission:

1. The Layer 2a headline now reads more precisely: "Mid-season HC change
   → pass-rate shift 1.36x p=0.002 on n=49 events" (was "HC change →
   pass-rate shift 1.36x" with implicit conflation of subsets).
2. A new Layer 2a' finding: "Offseason HC change → plays/game shift 1.17x
   p=0.021 on n=164 events." Different mechanism (pace), separate cell.
3. Reconciles with predictive validation: average offseason pass-rate
   shift is null, but conditional-on-prior-profile shift is forecastable.
   These are NOT contradictory — they describe different moments of the
   shift distribution.

### Downstream use

No change needed. production uses the **predictive validation slope** (0.30 HC,
0.45 OC), not the Layer 2a population mean. The slope is conditional on
prior profile and applies the direction-correct shift per coach. The
offseason null mean is consistent with this — slopes generate signed
adjustments that average to zero across the population if HCs come from
diverse priors, exactly as observed.

The production calibration is REINFORCED by this finding: if we had instead applied
"every offseason HC change shifts pass-rate by X" as a population-level
adjustment, we'd be wrong. The retread-conditioned slope handles the
direction correctly.

### Optional optional extension

Could add a **pace adjustment** for offseason HC changes:
`+1.17 / 1.0 = +17% pace shift magnitude` translated to per-position FP
via volume composition. Would affect WR/TE/RB equally (volume effect, not
mix). Modest signal (4.6 vs 3.9 plays/game absolute shift), and the
calibration isn't as clean as retread direction-conditioned slopes.
Defer pending propensity-matched design or LOSO calibration on extended
sample.

## Reproduce

```bash
python data/nfl/_src/hc_midseason_vs_offseason.py
```

Outputs:
- `data/nfl/hc_midseason_vs_offseason_results.csv` (6 rows: 3 metrics × 2 subsets)

## Design notes

**Mid-season window:** within-season, change-week from PFR `head_coach_games_split` field. Pre = first `change_after_games` weeks under prior HC, post = remaining weeks under interim.

**Offseason window:** K=6 cross-season. Pre = last 6 games of season N-1 under prior HC, post = first 6 games of season N under new HC. Excludes team-seasons that also had a mid-season change (cleans the natural experiment).

**Placebo:**
- Mid-season: stable team-seasons with random in-season split point (~700 (team-season, fake_split) pairs)
- Offseason: HC AND OC continuity team-seasons with same K=6 cross-season window (~347 events)

**Test:** Permutation B=5000 on mean |Δ| difference between real and placebo. Verdict thresholds: p<0.05 AND ratio>1.1 → "STAFF SHIFT MOVES NEEDLE"; p>=0.10 → "null"; in between → marginal.
