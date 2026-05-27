# DC-Change + OC Schematic-Layer + OC RZ-Focused Tests

**Date:** 2026-05-15
**Status:** Three follow-up tests complete. The single-locus HC-dominance hypothesis is **rejected** in favor of a **two-locus split** (HC owns macro philosophy, OC owns RZ fine-grain).

## Test 1 — DC-change event study (defensive symmetry)

Parallel design to OC-change event study, but for defensive coordinators
with HC continuity. Targets are defensive-allowed versions of pass-rate,
plays, RZ trips.

| Metric | n_real | n_plac | \|Δ\|_real | \|Δ\|_plac | ratio | p_perm | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| opp_pass_rate_residual | 144 | 428 | 0.0519 | 0.0542 | 0.96× | 0.731 | NO DETECTABLE SHIFT |
| opp_plays_residual | 144 | 428 | 4.00 | 4.10 | 0.98× | 0.633 | NO DETECTABLE SHIFT |
| opp_rz_trips_residual | 144 | 428 | 0.690 | 0.632 | 1.09× | 0.110 | NULL (trending) |

**Result:** Clean null on all three. DC changes do not shift opponent
offensive metrics significantly. Same pattern as OC test.

**Implication:** The coordinator-level null isn't offense-specific. Both
OC and DC are subordinate to HC at the generic outcome layer. RZ trips
trends marginally positive on both sides (OC 1.11×, DC 1.09×, both ~p=0.10)
— suggests a coordinator-level RZ signal that fragments below SNR threshold
at this resolution.

## Test 2 — OC schematic isolation (Layer 3 by role)

The Layer 3 within-game schematic test used `coaching_vector_id = HC|OC|DC`
composite and found COACH 15/20 cells dominant. This test decomposes that
composite by partitioning by each role independently.

### Mean F-statistic across 20 cells (5 buckets × 4 schematic styles)

| Role partition | Mean F |
|---|---:|
| HC\|OC composite | 6.47 |
| coach_full (HC\|OC\|DC) | 6.08 |
| **HC alone** | **5.43** |
| OC alone | 3.52 |
| DC alone | 3.27 |
| team_id baseline | 1.62 |

### Winner per cell (highest F among HC, OC, DC)

**HC wins 20/20 cells.** Every score-bucket × schematic-style combination
has HC as the dominant role partition.

| Style | HC wins | OC wins | DC wins |
|---|---:|---:|---:|
| pass_rate | 5/5 | 0 | 0 |
| shotgun_rate | 5/5 | 0 | 0 |
| no_huddle_rate | 5/5 | 0 | 0 |
| deep_throw_rate | 5/5 | 0 | 0 |

### Interpretation

The Layer 3 within-game COACH dominance lives at the **HEAD COACH** layer.
HC alone captures 84% of the joint HC|OC F (5.43 vs 6.47). OC and DC each
contribute modest secondary signal (~2× team baseline) but are clearly
subordinate.

The HC owns formation tendencies, tempo (huddle vs no-huddle), and
vertical-aggressiveness (deep-throw rate) as a **macro schematic identity**
that persists across coordinator changes.

## Test 3 — OC RZ-focused follow-up

The main OC-change test had RZ trips marginal (1.11×, p=0.075). This test
powers up with K=8 window and three RZ-specific schematic targets:

| Metric | n_real | n_plac | \|Δ\|_real | \|Δ\|_plac | ratio | p_perm | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| rz_pass_rate_residual | 175 | 393 | 0.0887 | 0.0806 | 1.10× | 0.075 | MARGINAL |
| **rz_shotgun_rate_residual** | **175** | **393** | **0.1024** | **0.0763** | **1.34×** | **0.0002** | **STAFF SHIFT MOVES NEEDLE** |
| rz_td_rate_residual | 175 | 393 | 0.1186 | 0.1202 | 0.99× | 0.565 | NO DETECTABLE SHIFT |

### Headline

**Red-zone shotgun-rate shifts 1.34× at OC changes, p=0.0002.**

This is the strongest p-value of any coaching-event study (HC pass_rate
was p=0.002, ten times weaker on the same scale). With sample size 175
real vs 393 placebo, signal-to-noise is unambiguous.

**The OC owns red-zone shotgun-rate.** When a new OC arrives even with
HC continuity, the red-zone formation choice (shotgun-empty vs under-center
heavy) shifts substantially.

### Why this didn't show in the main OC test

The main OC test used generic outcomes (pass_rate, plays, RZ trips count).
Pass-rate is HC-philosophy-locked; the new OC adapts to existing HC
preference and the magnitude doesn't shift. RZ trips count is field-state +
team-quality driven, not OC-driven.

But RZ FORMATION CHOICE is detailed OC game-planning. Goal-line packages,
heavy sets, motion concepts — these are OC-specific decisions that the
generic test wasn't asking the right question to detect.

## Combined three-test rewrite of the layered finding

The single-locus HC-dominance story collapses into a **two-locus split**:

| Locus | Owns | Evidence | Resolution |
|---|---|---|---|
| **HC** | Offensive philosophy (pass-rate magnitude) | Layer 2 HC: 1.36× p=0.002 | Within-season event |
| **HC** | General schematic identity (shotgun, no-huddle, deep-throw rates) | Layer 3 isolation: 20/20 cells | Within-game, score-bucketed |
| **OC** | Red-zone schematic fine-grain (RZ shotgun-rate) | OC RZ test: 1.34× p=0.0002 | Within-season event, field-position-conditional |
| **DC** | Not detected at outcome layer | DC test: null on opp metrics | Tested |
| **DC** | Untested at fine-grain (defensive RZ schematic) | — | Symmetric extension queued |

The HC owns the macro schematic vector; the OC owns red-zone fine-grain
schematic; the DC's locus is still unconfirmed at fine grain.

## Implications for the synthesis

The synthesis claim revises:

> "Coaching is the latent for schematic shifts at HC-change events"

becomes:

> "Coaching is the latent for schematic shifts, with a hierarchical
> two-locus structure:
> - HC owns macro schematic identity (formation, tempo, vertical aggressiveness)
>   and offensive pass-rate philosophy
> - OC owns red-zone schematic fine-grain (specifically RZ shotgun-rate)
> - DC has secondary partition signal at within-game resolution but no
>   detectable shift on opponent outcome metrics at coordinator-change events"

This is a **stronger paper claim** than the original single-locus version
because it identifies the SUBSTRATE for the coordinator-level signal that
the generic test missed.

## Implications for downstream use

The original recommendation ("flag HC changes, not OC changes") needs
revision:

- **For pass-rate / pass-game volume projections at W1-W4 of HC-change seasons:**
  apply HC-prior-schematic forecast (as originally planned)
- **For red-zone schematic projections** (which compose into receiving TD
  forecasts for slot WRs, TEs, and pass-catching RBs) **at W1-W4 of
  OC-change seasons:** apply OC-prior-RZ-shotgun-rate forecast

These are TWO event-triggered features, not one. Each gates on a different
predictive validation test.

## Test 4 — DC RZ-focused follow-up (added 2026-05-15)

Symmetric extension to OC RZ test. Targets are RZ-specific defensive metrics
where DC scheme could plausibly impose identity.

| Metric | n_real | n_plac | \|Δ\|_real | \|Δ\|_plac | ratio | p_perm | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| opp_rz_pass_rate_residual | 144 | 428 | 0.0787 | 0.0780 | 1.01× | 0.446 | NO DETECTABLE SHIFT |
| opp_rz_sack_rate_residual | 144 | 428 | 0.0522 | 0.0493 | 1.06× | 0.238 | NO DETECTABLE SHIFT |
| opp_rz_td_rate_residual | 144 | 428 | 0.1098 | 0.1190 | 0.92× | 0.851 | NO DETECTABLE SHIFT |

**Clean null on all three metrics.** Same K=8 window and methodology as OC RZ
test which carried at 1.34× p=0.0002. DC does NOT have a comparable
fine-grain locus.

### Coordinator-role asymmetry — the headline

| Role | Generic outcomes | RZ fine-grain |
|---|---|---|
| **OC** | NULL (0.97× p=0.636 on pass_rate) | **CARRIES (1.34× p=0.0002 on RZ shotgun-rate)** |
| **DC** | NULL (0.96× p=0.731 on opp_pass_rate) | NULL (1.01× p=0.446, 1.06× p=0.238, 0.92× p=0.851) |

The OC has a real authoring locus (RZ schematic fine-grain). The DC does
not — at any tested resolution.

### Two interpretations, both paper-worthy

1. **Defensive RZ identity is personnel-driven, not scheme-driven.** When
   the field shrinks to ≤20 yards, individual DL/LB/DB matchups dominate.
   DCs are reactive (adapting to opponent personnel/scheme) rather than
   impositional. The "DC fingerprint" we'd expect to see may be distributed
   across position-coach influence (DL coach + DB coach) rather than
   centralized in the DC role.

2. **Defensive coordination is HC-owned.** Defensive macro decisions
   sit with the HC just like offensive macro decisions do, while the OC
   has been carved out as an autonomous offensive authoring role. Testable
   via HC-change event study on opp_rz_X metrics.

Either interpretation makes the asymmetry itself the paper's headline
defensive finding.

## Open follow-up tests (re-prioritized after DC RZ null)

| # | Test | Rationale | Status |
|---|---|---|---|
| 1 | HC change × opp_rz_X (HC-defensive-locus test) | Distinguish interpretation 1 vs 2: does HC change shift RZ-defensive metrics? If yes, supports HC-owned defense. If no, supports personnel-driven. | Queued |
| 2 | Predictive validation: incoming OC's prior RZ shotgun rate → new team RZ shotgun shift | Falsification gate for the OC-RZ feature | Queued |
| 3 | Predictive validation: incoming HC's prior pass-rate philosophy → new team pass-rate shift | Falsification gate for the HC feature | Queued |
| 4 | Coach fingerprint clustering (HC + OC) | Visual deliverable + named style classes | Queued |
| 5 | Fantasy-outcome wiring (HC pass-rate + OC RZ-shotgun → FP residuals) | Connect findings to projection use case | Queued |
| 6 | Cross-league NBA mirror | Cross-domain replication | Queued |

## Reproduce

```bash
python data/nfl/_src/dc_change_event_study.py
python data/nfl/_src/oc_schematic_isolation.py
python data/nfl/_src/oc_rz_focused_followup.py
python data/nfl/_src/dc_rz_focused_followup.py
```

Outputs:
- `data/nfl/dc_change_event_summary.csv`, `dc_change_event_panel.csv`
- `data/nfl/oc_schematic_isolation_results.csv`
- `data/nfl/oc_rz_focused_summary.csv`, `oc_rz_focused_panel.csv`
- `data/nfl/dc_rz_focused_summary.csv`, `dc_rz_focused_panel.csv`
