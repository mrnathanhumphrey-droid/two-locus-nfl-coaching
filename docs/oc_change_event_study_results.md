# OC-Change Event Study — Coordinator-Level Replication of Layer 2

**Date:** 2026-05-15
**Pre-registered hypothesis:** HC-change pass-rate effect replicates at the OC (offensive coordinator) level.
**Outcome:** **HYPOTHESIS REJECTED.** Effect is HC-specific, not coordinator-general.

## Headline

OC changes do NOT shift pass_rate residuals year-over-year, despite a sample
3.2× larger than the original HC test. The HC-change pass-rate effect is
organizational-leader-specific — the new HC's philosophy, not the new OC's
play-calling, drives the shift.

| Metric | n_real | n_plac | \|Δ\|_real | \|Δ\|_plac | ratio | p_perm | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| pass_rate_residual | 159 | 358 | 0.0588 | 0.0605 | **0.97×** | 0.636 | NO DETECTABLE SHIFT |
| plays_residual | 159 | 358 | 3.73 | 3.96 | 0.94× | 0.794 | NO DETECTABLE SHIFT |
| red_zone_trips_residual | 159 | 358 | 0.657 | 0.594 | 1.11× | 0.075 | MARGINAL |

## HC vs OC side-by-side

| Metric | HC change (n=49) | OC change (n=159) |
|---|---|---|
| pass_rate_residual | **1.36× p=0.002 (carries)** | 0.97× p=0.636 (null) |
| plays_residual | 1.10× p=0.187 (null) | 0.94× p=0.794 (null) |
| red_zone_trips_residual | 1.07× p=0.256 (null) | 1.11× p=0.075 (marginal) |

The single bar that carried at HC level (pass_rate) is the bar that goes
NULL at OC level — even with 3.2× the sample size. The single bar that
shows directional signal at OC level (RZ trips) was null at HC level.

## Design

### Treatment events: offseason OC change with HC continuity
For team-season (T, S) with prior-year (T, S-1) record:
- HC[T, S] == HC[T, S-1] (HC unchanged, isolates OC effect)
- OC[T, S] != OC[T, S-1] (OC changed)
→ 175 candidate events, 159 with computable pre/post windows.

### Placebo events: continuous staff
For team-season (T, S) with prior-year (T, S-1) record:
- HC[T, S] == HC[T, S-1]
- OC[T, S] == OC[T, S-1]
→ 393 candidate events, 358 with computable pre/post windows.

### Window definition
- Pre = mean residual over last K=6 games of season S-1
- Post = mean residual over first K=6 games of season S
- \|Δ\| = \|mean(post) - mean(pre)\|

K=6 chosen to be consistent with the HC test scale (HC test used full-season
pre/post within one season, ~5-13 games each side). For OC offseason design,
K=6 gives 1/3 season per window with adequate power.

### Permutation null
B = 5000. Shuffle real/placebo labels, compute mean(\|Δ\|) difference, count
P(null ≥ observed).

## Sample size advantage over HC test

| | HC test (Layer 2) | OC test |
|---|---:|---:|
| treatment events | 49 | 159 |
| placebo events | 700 (random fake splits in stable team-seasons) | 358 (continuous staff) |
| temporal unit | within-season | cross-season |

OC test has **3.2× more treatment events** and **comparable** placebo power.
The negative result is power-robust — this isn't a sample-size failure.

## Interpretation

### The pass-rate latent is HC-specific

The cleanest interpretation: the HC owns the offensive philosophy decision
("what kind of football do we play") and selects/dismisses OCs through that
filter. A new OC under continuing HC adapts to the existing system rather
than imposing their own.

This is structurally consistent with the Layer 3 finding that
coaching_vector_id (which is `HC|OC|DC` joined) loads at sub-snap resolution
on schematic identity. The within-game COACH partition wins on shotgun /
no-huddle / deep-throw — but the coaching_vector_id is dominated by HC
identity, not OC identity, given league norms of HC tenure stability.

### What OCs DO own (tentative)

Red-zone trips ratio 1.11× at p=0.075 is the only directional signal. The
mechanism is plausible — red-zone game-planning is detailed OC work
involving formation-specific play menus, hot-route concepts, etc. But the
signal sits below the SNR discipline threshold (p < 0.05) and we don't
claim it.

A follow-up RZ-specific test with:
- RZ-pass-rate as target instead of generic RZ trips
- Per-RZ-play schematic breakdown (formation, motion, pre-snap shift)
- Larger window (K=8 or K=10) for power

could potentially upgrade this from marginal to carries, OR confirm null.

### What OCs do NOT own (this study confirms)

- **Pass-rate magnitude.** This is HC-philosophy + roster, not OC play-calling.
- **Pace (plays/game).** Multiple lines of evidence (Layer 1 null on plays
  for coach_partition, Layer 2 HC null on plays_residual, OC test null) all
  converge on: pace is team identity, not coaching at any level.

## Implications for the synthesis

The four-layer synthesis claim sharpens:

> "Coaching IS the latent for schematic shifts at HC-change events"

becomes:

> "Coaching IS the latent for schematic shifts at HC-change events. The
> effect is HEAD-COACH-specific, not coordinator-general. OC changes under
> continuing HC produce no detectable pass-rate shift even at 3× sample size."

For the production-integration discussion (event-triggered feature for
teams with coordinator changes), this is a sharper recommendation:
**flag HC changes, not OC changes.** OC turnover under continuing HC has
no detectable forward signal at the metrics tested here.

## What the negative result rules out

- **"Coaching staff change = schematic shift" (the broad claim).** REJECTED.
  Not all coaching changes carry the same signal.
- **"OC owns pass-rate" intuition.** REJECTED. OCs execute within an HC's
  pass-rate philosophy.
- **"Sample-size limited HC test missed coordinator-level signal" worry.**
  REJECTED. OC test had ~3× sample size and produced an even cleaner null.

## What this does NOT rule out

- **DC-change effects on defensive metrics.** Different role, different
  outcome surface. Untested here.
- **Joint HC+OC change effects.** When the HC is fired AND the OC departs,
  both move at once and we'd capture it via the HC test. The 168
  HC_CHANGE_EXCLUDE events in our panel had HC change; many also had OC
  change. These are excluded from this test because the OC effect can't
  be isolated.
- **Schematic-specific OC effects.** This test used generic outcome
  metrics (pass_rate, plays, RZ trips), not within-game schematic styles
  (shotgun, no-huddle, deep-throw). The Layer 3 within-game test used
  coaching_vector_id (HC+OC+DC composite). An OC-only schematic test
  (varying OC, fixed HC+DC) would isolate whether the schematic-style
  loadings live at HC or OC level. Open experiment.
- **RZ-specific OC effects.** The p=0.075 marginal hint is worth a
  targeted RZ-focused follow-up.

## Open follow-up tests

1. **DC-change × DEFENSIVE metrics**: same design, defensive coordinator
   changes, target = opponent EPA/play, pass yards allowed, opponent
   pass-rate residual. Tests whether the HC-as-locus claim symmetrizes
   to the defensive side, OR whether DCs do show their own latent.

2. **Schematic-style isolation at OC level**: Layer 3 test but partition by
   OC_id only, with HC_id + DC_id held constant. Tests whether the within-
   game schematic loadings (shotgun, no-huddle, deep-throw) come from the
   HC or the OC within the coaching_vector_id composite.

3. **RZ-focused OC follow-up**: same OC-change event design, but target =
   RZ pass-rate, RZ scoring rate, RZ TD rate. Power-up the marginal signal.

## Reproduce

```bash
python data/nfl/_src/oc_change_event_study.py
```

Outputs:
- `data/nfl/oc_change_event_panel.csv` (1,551 rows: 517 events × 3 metrics)
- `data/nfl/oc_change_event_summary.csv` (3 rows, one per metric)
