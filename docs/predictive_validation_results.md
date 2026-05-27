# Predictive Validation — HC + OC Portability

**Date:** 2026-05-15
**Status:** Both HC and OC predictive validation PASS. Falsification gates open for both event-triggered features. Schematic identity is PORTABLE — incoming HC/OC prior profile forecasts the new-team shift with significant signal.

## Headline

| Test | n_events | Pearson r | p_one | OLS slope | OLS R² | Dir agree | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| HC pass-rate portability | 41 | +0.386 | 0.0044 | 0.30 | 0.149 | 59% | **PORTABLE** |
| OC RZ-shotgun portability | 149 | **+0.483** | **0.0000** | **0.45** | **0.234** | **68%** | **PORTABLE** |

Both tests pass the SNR-discipline gate (p<0.05, r>0.2). **The OC predictive
validation is stronger than HC across every metric**, consistent with the
mechanism that OCs author specific RZ schematic playbooks they bring with
them, while HCs bring broader philosophy that's more roster-adaptive.

## Design

For each "retread event" (coach who moved between teams within our data
window), compute three profiles:

- `profile_coach_prior`: the coach's metric average at their PRIOR team
  (last K games of last season at that team)
- `profile_new_pre`: the new team's metric average UNDER PRIOR STAFF
  (last K games of season before the new coach arrives)
- `profile_new_post`: the new team's metric average UNDER NEW COACH
  (first K games of new season)

Observed shift: `shift_obs = profile_new_post - profile_new_pre`
Predicted shift: `shift_pred = profile_coach_prior - profile_new_pre`

The prediction encodes "the new coach should pull the new team toward
their own prior profile." Pearson r(shift_pred, shift_obs) measures
forecastability. OLS slope measures calibration (1.0 = perfect transfer;
0 = no transfer; intermediate = partial portability).

Null: permutation test on `shift_pred` labels (B=5000). P-value =
fraction of permutations with r ≥ observed.

### Why this design isolates portability

The baseline `profile_new_pre` controls for team-roster fixed effects
already in the new team's prior performance. The prediction `shift_pred`
asks specifically: among the gap between (coach's prior profile) and (new
team's profile), how much closes?

If 100% closes → slope 1.0 (perfectly portable coach identity)
If 0% closes → slope 0.0 (team-adapted; coach doesn't transfer)
Intermediate slopes mean the coach's identity is partially portable + the
roster/team partially shapes the new profile.

## HC predictive validation — n=41

### Hit list (top 5 by absolute observed shift)

| HC | Move | Predicted | Observed |
|---|---|---:|---:|
| Andy Reid | PHI 2012 → KC 2013 | +0.185 | +0.230 |
| Frank Reich | IND 2022 → CAR 2023 | +0.123 | +0.163 |
| Jim Mora | ATL 2006 → SEA 2009 | ~0 | +0.135 |
| Pat Shurmur | CLE 2012 → NYG 2018 | +0.019 | +0.108 |
| Tony Dungy | TB 2001 → IND 2002 | +0.022 | +0.097 |

### Miss list (largest negative observed shifts)

| HC | Move | Predicted | Observed | Note |
|---|---|---:|---:|---|
| John Fox | DEN 2014 → CHI 2015 | +0.036 | -0.133 | Direction flip |
| Tom Coughlin | JAX 2002 → NYG 2004 | -0.123 | -0.104 | Well-predicted |
| Rex Ryan | NYJ 2014 → BUF 2015 | -0.204 | -0.091 | Magnitude undershoot |
| Chip Kelly | PHI 2015 → SF 2016 | +0.056 | -0.075 | Direction flip — roster mismatch (failed pace experiment) |
| Eric Mangini | NYJ 2008 → CLE 2009 | +0.112 | -0.060 | Direction flip |

The direction flips are interesting mechanism cases — Chip Kelly to SF
is the canonical "failed style transfer" where the receiver/QB room couldn't
support his pace concepts. These misses are not random noise; they're
roster-imposed limits on portability.

## OC predictive validation — n=149

### Hit list (top 5 by absolute observed shift)

| OC | Move | Predicted | Observed |
|---|---|---:|---:|
| Curtis Modkins | BUF 2012 → SF 2016 | +0.215 | +0.453 |
| Rob Chudzinski | CLE 2008 → CAR 2011 | +0.011 | +0.383 |
| Jim Bob Cooter | DET 2018 → IND 2023 | +0.064 | +0.332 |
| Jimmy Raye | LV 2005 → SF 2009 | +0.110 | +0.318 |
| Greg Olson | LA 2007 → TB 2009 | +0.026 | +0.278 |

### Miss list (largest negative observed shifts)

| OC | Move | Predicted | Observed |
|---|---|---:|---:|
| Gary Kubiak | DEN 2005 → BAL 2014 | -0.292 | -0.640 |
| Rick Dennison | DEN 2016 → BUF 2017 | -0.361 | -0.386 |
| Chan Gailey | KC 2008 → NYJ 2015 | +0.085 | -0.323 |
| Darrell Bevell | MIN 2010 → SEA 2011 | -0.200 | -0.312 |
| Mike Shula | TB 1999 → CAR 2013 | -0.239 | -0.298 |

Most of the OC "miss list" entries actually show **strong direction
prediction** (Kubiak predicted -0.29, observed -0.64 — same sign, just
magnitude undershoot). The OC's prior profile gets the direction right
~68% of the time and the magnitude on average ~45% of the way.

## Why OC > HC on portability

The 45% OC calibration vs 30% HC calibration is structurally consistent
with the broader finding:

- **HC** owns macro philosophy (pass-rate magnitude, schematic identity).
  This is a HIGH-LEVEL bundle that interacts strongly with roster
  composition. A HC who wants to throw a lot needs receivers and a QB
  who can execute that; if the new team's roster doesn't support it,
  the macro profile doesn't transfer cleanly.

- **OC** owns RZ schematic fine-grain (formation choice in the red zone).
  This is a LOW-LEVEL playbook decision that's more roster-independent.
  Whether you go shotgun-empty or under-center-heavy in the red zone is
  more about the OC's preferred concepts than about which RB or WR is on
  the roster.

The two-locus finding now extends: HC = high-level philosophy with
moderate portability; OC = low-level playbook with high portability. The
COACHING resolution and the PORTABILITY both differ by role.

## Implications

### Paper

The forecastability section is the natural capstone. It establishes:
1. Coaching latents are real (positive partition F-stats + event shifts)
2. They're decomposable by role (HC macro + OC RZ fine-grain, structural defensive null)
3. They're predictable from coach prior data (HC slope 0.30, OC slope 0.45)
4. The schematic latent is genuinely portable but partially roster-adapted

This is "and we can forecast it" — the move from descriptive finding to
predictive science.

### Downstream use

Both event-triggered features have green light:

| Feature | Trigger | Forecast | Apply window |
|---|---|---|---|
| HC pass-rate shift | offseason HC change with prior HC tenure | `+slope_HC × (HC_prior_pass_rate - new_team_pre_pass_rate)` where slope_HC=0.30 | W1-W4 of new season |
| OC RZ-shotgun shift | offseason OC change with prior OC tenure | `+slope_OC × (OC_prior_rz_shotgun - new_team_pre_rz_shotgun)` where slope_OC=0.45 | W1-W6 of new season |

For teams with first-time HC or OC (no prior tenure in our window), the
feature defaults to no adjustment — trailing rate-form baseline handles those
correctly via the cold-start prior.

### What the slopes mean numerically

A predicted shift of +0.10 in pass_rate (HC) means the prior HC tenure
ran 10pp higher pass-rate than the new team. With slope 0.30, the model
forecasts the new team will shift +0.03 pass-rate W1-W4. This composes
into per-player downstream consumers via target-share decomposition.

A predicted shift of +0.15 in RZ shotgun-rate (OC) means the prior OC
tenure ran 15pp higher RZ shotgun-rate than the new team. With slope 0.45,
the model forecasts +0.0675 RZ shotgun-rate shift W1-W6. This composes
into receiving-TD quantities for relevant positional groupings.

## Reproduce

```bash
python data/nfl/_src/hc_predictive_validation.py
python data/nfl/_src/oc_predictive_validation.py
```

Outputs:
- `data/nfl/hc_predictive_validation_paired.csv` (41 retread HC events)
- `data/nfl/hc_predictive_validation_summary.csv`
- `data/nfl/oc_predictive_validation_paired.csv` (149 retread OC events)
- `data/nfl/oc_predictive_validation_summary.csv`
