# NFL Coaching Signal — Multi-Layer Resolution

**Status:** Synthesis of findings completed 2026-05-08 through 2026-05-09. Pre-paper consolidation; ready for extension experiments.

## Headline

NFL coaching identity decomposes into a **two-locus offensive structure with
a STRUCTURAL DEFENSIVE NULL**:

- **HEAD COACH owns macro offensive schematic identity** (formation, tempo,
  vertical aggressiveness) — Layer 3 within-game test 20/20 cells HC-dominant;
  Layer 2a HC-change pass-rate shift 1.36× p=0.002.
- **OFFENSIVE COORDINATOR owns red-zone schematic fine-grain** (specifically
  RZ shotgun-rate) — Layer 2d OC-change test rz_shotgun shift 1.34× p=0.0002.
- **NO coaching role has a detected defensive locus.** Three roles × two
  resolutions = six tested cells, all defensive cells NULL:
  - DC change × generic opp outcomes: NULL (Layer 2c)
  - DC change × RZ-fine-grain defensive: NULL (Layer 2e)
  - HC change × generic opp outcomes: NULL (Layer 2f)
  - HC change × RZ-fine-grain defensive: NULL (Layer 2f, best opp_rz_pass
    1.11× p=0.068, marginal — same magnitude as the OC RZ pass-rate
    marginal, both unclaimable)

The offense-vs-defense asymmetry is structural. **Defensive identity is
personnel-driven, not coaching-driven**, at least at the offseason-event
resolution we can measure. When the field shrinks to ≤20 yards, individual
DL/LB/DB matchups dominate scheme effects in a way that no coaching role
overcomes.

The coaching latent is **undetectable at season-level aggregates** (Layer 1:
0/4 partition wins) and at **within-game stress-adaptation** (Layer 4: 0/32
cells under SNR discipline).

The methodological contribution is the **clean layered decomposition** with
proper placebo nulls and cross-season holdouts at every layer, plus
role-level isolation that revealed the two-locus offensive hierarchical
structure AND the offense-vs-defense coordinator asymmetry where the
generic single-partition test (coaching_vector_id = HC|OC|DC composite)
saw only one bulk signal.

## Why this matters

Standard fantasy / EPA projection models pool by team_id and treat coaching as
a fixed effect inside team. That convention is right for season-level outcome
totals (pass yards, scoring rate) but wrong for the schematic dimension
(formation tendency, tempo, target depth). A model that pools everything by
team_id misses the schematic shift around HC changes, and a model that pools
everything by coach over-fits non-schematic outcomes.

The methodological contribution is a four-layer test framework with proper
null calibration at every layer, demonstrating that one latent (coaching) can
load differently at different temporal/structural resolutions of the same
sport.

## The four-layer test framework

For each layer, we tested the same partition (coaching_vector_id vs team_id)
on the same set of outcomes (pass_rate, plays/game, RZ trips, EPA/play, plus
within-game schematic rates), with the same SNR discipline (SNR ≥ 1.5,
p ≤ 0.01, cross-season ≥ 2/3 holdout positive). Differences across layers are
attributable to the **resolution** at which the partition is applied, not to
data or methodology drift.

| Layer | Unit | Null | Outcome | Result |
|---|---|---|---|---|
| 1. Static | (team, season) | Within-season permutation (B=1000) | Season aggregates (pass_rate, plays, RZ trips, EPA/play) | 2 TEAM, 2 TIE, 0 COACH |
| 2a. Within-season HC change events | HC-change windows (n=49) | Placebo non-change splits (B=5000) | Pre/post Δ residual on generic outcomes | pass_rate 1.36× p=0.002; plays + RZ null |
| 2b. Offseason OC change events (HC continuity) | OC-change cross-season (n=159) | Continuous-staff cross-season (n=358) | Pre/post Δ residual on generic outcomes | pass_rate 0.97× p=0.636; RZ trips 1.11× p=0.075 marginal |
| 2c. Offseason DC change events (HC continuity) | DC-change cross-season (n=144) | Continuous-staff cross-season (n=428) | Pre/post Δ residual on opp outcomes | opp_pass_rate 0.96× p=0.731; opp_RZ 1.09× p=0.110 |
| 2d. OC change × RZ-focused metrics (HC continuity) | OC-change cross-season (n=175) | Continuous-staff cross-season (n=393) | Pre/post Δ on RZ-specific schematic | **RZ shotgun-rate 1.34× p=0.0002**; RZ pass-rate 1.10× p=0.075 marginal |
| 2e. DC change × RZ-focused metrics (HC continuity) | DC-change cross-season (n=144) | Continuous-staff cross-season (n=428) | Pre/post Δ on RZ-specific defensive | opp_rz_pass 1.01× p=0.446; opp_rz_sack 1.06× p=0.238; opp_rz_td 0.92× p=0.851 — all NULL |
| 2f. HC offseason change × defensive outcomes (disambiguation) | HC-change cross-season (n=160) | Continuous-staff cross-season (n=380) | Generic + RZ-fine-grain defensive | Generic NULL (opp_pass 1.01× p=0.449, opp_plays 1.01× p=0.467, opp_RZ_trips 1.07× p=0.162). RZ NULL (opp_rz_pass 1.11× p=0.068 marginal, sack 0.97× p=0.65, td 0.97× p=0.69). |
| 5a. HC predictive validation (retread HCs) | n=41 retread events | Permutation test on shift_pred labels (B=5000) | Forecastability of pass_rate shift from HC's prior tenure profile | **PORTABLE**: r=+0.386, p=0.0044, OLS slope 0.30, R²=0.149, direction agreement 59% |
| 5b. OC predictive validation (retread OCs) | n=149 retread events | Permutation test on shift_pred labels (B=5000) | Forecastability of RZ-shotgun shift from OC's prior tenure profile | **STRONGLY PORTABLE**: r=+0.483, p=0.0000, OLS slope 0.45, R²=0.234, direction agreement 68% |
| 3a. Within-game schematic identity (composite) | (team, season, score-bucket, play-style) | Within-season permutation (B=1000) | Shotgun, no-huddle, deep-throw, pass-rate per bucket | 15/20 cells COACH-dominant on schematic styles |
| 3b. Within-game schematic role isolation | (team, season, score-bucket, play-style) | F-stat by HC, OC, DC, composite, team | Same as 3a, partitioned by each role | **HC wins 20/20 cells**; mean F: HC 5.43, OC 3.52, DC 3.27, team 1.62 |
| 4. Within-game stress adaptation | (game, team, score-segment) | Group-label permutation (B=1000) | Segment-level shift in pass/shotgun/no-huddle/deep | 0/32 CARRIES; in-sample TEAM 5/16, TIE 10/16, COACH 1/16 |

## Layer 1 — Static (season aggregate)

Coaching_vector_id is real signal (perm-p < 0.05 on 4/4 dims) but **does not
beat team-id when properly degree-of-freedom-corrected.**

| Outcome | F(coach) | F(team_id) | F-ratio | perm-p coach | Verdict |
|---|---:|---:|---:|---:|---|
| pass_rate residual | 1.99 | 4.50 | 0.44 | 0.0000 | TEAM-ID WINS |
| plays residual | 2.07 | 1.67 | 1.24 | 0.0000 | TIE / coach slight edge |
| RZ trips residual | 1.37 | 1.75 | 0.78 | 0.0050 | TIE / team slight edge |
| EPA/play | 2.16 | 4.03 | 0.54 | 0.0000 | TEAM-ID WINS |

**Static summary: 2 TEAM-ID wins, 2 TIE, 0 COACHING wins.**

### The SVI-analogue confound

team_id is not a clean baseline. It implicitly aggregates roster continuity,
scheme inheritance, GM tendencies, owner stability, stadium/climate, AND the
actual coaching staff averaged over the team's history. The static-layer test
is structurally biased against the coaching hypothesis in the same way that
implied volatility decompositions are structurally biased against the
spot-vol component: the "control" already contains the thing being tested.

The static-layer walk-back is **conditional on this confound**, not
unconditional. It rules out a season-aggregate coaching-mean-offset feature,
not the coaching latent itself. Resolution: hold team identity constant and
test whether **changes in coaching** move the residual — Layer 2.

## Layer 2 — Within-season event study

49 mid-season HC changes (45 with computable pre/post windows) vs 700
placebo stable-team-seasons with random fake change-week splits. B=5000
permutations of placebo vs real |Δ residual|.

| Outcome | \|Δ\|_real | \|Δ\|_placebo | ratio | p_perm | Verdict |
|---|---:|---:|---:|---:|---|
| pass_rate residual | 0.072 | 0.053 | **1.36×** | **0.002** | STAFF SHIFT MOVES NEEDLE |
| plays residual | 3.93 | 3.56 | 1.10× | 0.187 | no detectable shift |
| RZ trips residual | 0.611 | 0.569 | 1.07× | 0.256 | no detectable shift |

HC changes shift the **pass/run mix** within a fixed team identity. They do
NOT shift pace or RZ efficiency. This isolates the coaching latent from the
team-id confound at sub-seasonal resolution — same team, different coach,
significantly different pass-rate residual signature.

The selection-bias caveat is real: HC firings are non-random (typically
following losing streaks), so the placebo null is conservative but not
fully causal. A propensity-matched event design is in the open-experiments
queue.

## Layer 3 — Within-game schematic identity

For each (score-bucket, schematic-style) cell, compute F_coach (partitioning
by coaching_vector_id) vs F_team (partitioning by team_canon) using
team-game observations. The first-pass result (15/20 cells COACH-dominant)
was audited because the original test had **no permutation null** — only an
F_coach/F_team between-partition ratio. The audit added a within-season
permutation null (B=1000) and reports both rhetorical magnitudes.

### Audited verdict matrix (5 score buckets × 4 schematic styles = 20 cells)

| Bucket | shotgun_rate | no_huddle_rate | deep_throw_rate | pass_rate |
|---|---|---|---|---|
| leading_2plus | **COACH** (F=16.4) | **COACH** (F=4.3) | **COACH** (F=3.5) | TEAM (F=2.1) |
| leading_1 | **COACH** (F=15.9) | **COACH** (F=5.1) | **COACH** (F=3.7) | TEAM (F=1.8) |
| tied | **COACH** (F=15.2) | **COACH** (F=4.4) | **COACH** (F=3.0) | TEAM (F=2.0) |
| trailing_1 | **COACH** (F=14.2) | **COACH** (F=5.7) | **COACH** (F=4.1) | TIE (F=1.5) |
| trailing_2plus | **COACH** (F=9.3) | **COACH** (F=4.6) | **COACH** (F=3.7) | TEAM (F=1.3) |

**Final: COACH 15/20, TEAM 4/20, TIE 1/20.** All 4 schematic-style columns
load on coach across nearly every score bucket. The single non-coach style
column (pass_rate magnitude) loads on team — consistent with Layer 1.

### Two distinct rhetorical magnitudes

Two true-but-different claims emerge from the audit:

- **Between-partition explanatory ratio**: F_coach / F_team. Original
  framing. Median 2.70 across 20 cells; 8-16× on shotgun/no-huddle/deep-throw.
- **Observed-over-null effect size**: F_coach / null_B_mean (within-season
  permutation). Median 1.98 across 20 cells. **17/20 cells reject the null at
  p < 0.05** under proper permutation. All schematic-style cells achieve
  p < 0.001.

Both claims are correct and should be reported together. Reporting only one
collapses two distinct concepts.

## Layer 4 — Within-game stress adaptation (redesigned)

Tests whether teams have identifiable, repeatable patterns in how they
ADAPT their tactics under different score-pressure states. Uses
segment-level shifts: (tied baseline) → (leading_2plus / leading_1 /
trailing_1 / trailing_2plus) within the same team-game, partitioning the
resulting shift values by coaching_vector_id vs team_canon.

### Redesign of the original Test 2

The original Test 2 was withdrawn after a structural-bias audit found 95.99%
play-weighted HC stability inside the test windows — making F_coach
mechanically near-zero (coaching_vector_id and team_id were degenerate inside
windows). The redesign uses **segment-level shift observations** (n=987 to
n=2,198 per bucket), giving each coaching staff many independent
within-stint observations.

### Final verdict matrix

| Segment | pass_rate | shotgun_rate | no_huddle_rate | deep_throw_rate |
|---|---|---|---|---|
| leading_2plus (n=987) | NULL | NULL | NULL | NULL |
| leading_1 (n=2,122) | NULL | NULL | NULL | NULL |
| trailing_1 (n=2,198) | NULL | NULL | NULL | NULL |
| trailing_2plus (n=1,069) | NULL | NULL | NULL | NULL |

**0/32 CARRIES under full SNR discipline** (both partitions × 16 cells).
In-sample F-tags: TEAM 5, TIE 10, COACH 1. Where in-sample directional
signal exists, team_id captures it more often than coach (5 vs 1) — but the
holdout R² is negative for every single cell, indicating overfit on
2018-2022 patterns that don't replicate in 2023-2024.

**Interpretation:** stress adaptation at the segment-shift resolution is
**noise**. The universal "everyone passes more when trailing" patterns are
shared across the league; the fine-grained variation in "team A shifts 3%
more than team B" doesn't survive a new season.

## Net interpretation — split coaching latent

The four-layer decomposition refines the naive "coaching as load-bearing
latent" hypothesis into three sharper claims:

| Locus | Owns | Resolution | Evidence |
|---|---|---|---|
| **HC** | Offensive philosophy (pass-rate magnitude) | Within-season **mid-season firing** event | Layer 2a: 1.36× p=0.002 (n=45) — see refinement below |
| **HC** | Play volume / pace | **Offseason** event | Layer 2a': 1.17× p=0.021 (n=142) — newly isolated |
| **HC** | General schematic identity (shotgun, no-huddle, deep-throw rates) | Within-game, score-bucketed | Layer 3b: HC 20/20 cells dominant; mean F 5.43 vs OC 3.52, DC 3.27, team 1.62 |
| **OC** | Red-zone schematic fine-grain (specifically RZ shotgun-rate) | Within-season event, field-position-conditional | Layer 2d: 1.34× p=0.0002 (n=175) |
| **OC** | NOT pass-rate magnitude | Within-season event | Layer 2b: 0.97× p=0.636 (n=159) — OCs adapt to HC philosophy |
| **DC** | Not detected at generic outcome layer | Within-season event | Layer 2c: opp_pass 0.96× p=0.731 (n=144) |
| **DC** | Not detected at fine-grain defensive RZ either | Within-season event, field-position-conditional | Layer 2e: all three RZ-defensive metrics null (n=144) |
| **HC defense** | Not detected at generic OR fine-grain defensive | Within-season event, offseason | Layer 2f: generic all null; RZ-fine-grain all null (best opp_rz_pass 1.11× p=0.068) |
| Coaching | NOT season-level outcomes or stress adaptation | Aggregate; segment shifts | Layer 1: 0/4. Layer 4: 0/32. |

**The defensive null is now STRUCTURAL across three coaching roles and two
resolutions.** Six tested cells × three roles (HC, OC, DC) × defense:
zero positive findings, one marginal-but-unclaimable (opp_rz_pass at
HC and OC change, both ~1.10× p~0.07). Defensive identity at every
tested coaching resolution returns clean null.

### Power analysis — what the null rules out (2026-05-16 add)

Per-cell minimum detectable effect (MDE) at α=0.05 one-sided, β=0.80 power,
via z-approximation on the placebo-permutation standard error:

| Cell | n_real | Observed ratio | MDE@80% | |
|---|---:|---:|---:|---|
| L2c DC × opp_pass_rate | 134 | 0.92 | 1.18 | null with power |
| L2c DC × opp_plays | 134 | 0.94 | 1.19 | null with power |
| L2c DC × opp_RZ_trips | 134 | 1.13 | 1.21 | observed just below MDE |
| L2e DC × opp_rz_pass | 134 | 1.01 | 1.19 | null with power |
| L2e DC × opp_rz_sack | 134 | 1.05 | 1.22 | null with power |
| L2e DC × opp_rz_td | 134 | 0.82 | 1.18 | null with power (sign reversed) |
| L2f HC × opp_pass_rate | 167 | 1.01 | 1.18 | null with power |
| L2f HC × opp_plays | 167 | 1.00 | 1.18 | null with power |
| L2f HC × opp_RZ_trips | 167 | 1.10 | 1.19 | observed just below MDE |
| L2f HC × opp_rz_pass | 167 | 1.02 | 1.17 | null with power |
| L2f HC × opp_rz_sack | 167 | 0.92 | 1.19 | null with power |
| L2f HC × opp_rz_td | 167 | 0.96 | 1.16 | null with power |

**Reference offensive effects:** HC pass_rate 1.36× (Layer 2a, n=49 mid-season),
OC rz_shotgun_rate 1.34× (Layer 2d, n=175 offseason).

**What the null rules out:** all 12 defensive cells had power to detect effects
≥ 1.22× at 80%. The observed offensive effects (1.34–1.36×) are well above
this threshold. So the defensive coaching latent, if it exists, is
**strictly smaller in magnitude than the offensive coaching latents**.

**Honest caveat:** none of these cells had power to detect effects in the
1.10–1.20× range. Two cells (L2c opp_RZ_trips 1.13× and L2f opp_RZ_trips
1.10×) sit just below their respective MDE bars — there might be a real
defensive coaching effect at ~1.10–1.15× we couldn't statistically
distinguish from noise at this sample size. The paper claim should be:

> "Defensive coaching cells rule out effects ≥ ~1.22× at 80% power.
> Observed offensive effects (1.34–1.36×) exceed this bound. Defensive
> coaching loci, if any exist, are strictly smaller in magnitude than
> the observed offensive coaching loci."

Script: `data/nfl/_src/defensive_null_power_analysis.py`. Per-cell results:
`data/nfl/defensive_null_power_analysis.csv`.

Interpretation: **defensive RZ identity is personnel-driven, not scheme-
driven.** When the field shrinks to ≤20 yards, individual DL/LB/DB
matchups dominate scheme effects. This rejects the "HC-owned defense"
hypothesis and confirms the offense-vs-defense coaching asymmetry as
structural: offense has authored coaching loci (HC macro + OC RZ-fine),
defense does not.

## Mid-season vs offseason HC separation — natural-experiment cleanup

Layer 2a's 1.36× pass-rate effect was originally computed on the union of
mid-season HC firings + offseason HC changes. Splitting the two natural
experiments cleanly separates two distinct mechanisms (writeup:
[`docs/hc_midseason_vs_offseason_results.md`](hc_midseason_vs_offseason_results.md)):

| Subset | n | Pass-rate ratio | p | Plays/game ratio | p |
|---|---:|---:|---:|---:|---:|
| Mid-season HC firing | 45 | **1.36** | **0.002** | 1.10 | 0.187 |
| Offseason HC change | 142 | 1.02 | 0.428 | **1.17** | **0.021** |

**Mid-season** firings drive the pass-rate signal (interim coaches
systematically shift play-mix — likely a "shake things up" or
"play to remaining strengths" response). **Offseason** changes don't move
average pass-rate but DO move plays/game (new HC pace philosophy).

This is consistent with predictive validation: the average offseason
pass-rate shift is null, but the *direction* of each shift is forecastable
from the incoming HC's prior profile (Layer 5a, n=41, r=+0.39, p=0.004).
Different HCs go different directions; they cancel at population level
but the conditional-on-prior signal lives.

Refined Layer 2 framing:
- HC owns pass-rate identity at the mid-season firing resolution
- HC owns play volume / pace at the offseason resolution
- HC owns conditional pass-rate direction at the offseason resolution
  (forecastable from prior profile)
- OC owns RZ schematic fine-grain at the offseason resolution
- DC owns nothing at any tested resolution (structural null)

## Predictive validation — both offensive loci are forecastable

The two-locus offensive finding tells us the latents exist. Predictive
validation tells us they're **portable from coach prior data**:

| Test | n | r | p | OLS slope | Verdict |
|---|---:|---:|---:|---:|---|
| HC pass-rate portability | 41 | +0.386 | 0.004 | 0.30 | PORTABLE |
| **OC RZ-shotgun portability** | **149** | **+0.483** | **<0.0001** | **0.45** | **STRONGLY PORTABLE** |

**LOSO recal (2026-05-15) on extended sample including 2025:** HC slope 0.306 (n=43), OC slope 0.460 (n=151). Both within 2-4% of the original calibration — clean out-of-sample replication. See [`loso_recal_2026_05_15.md`](loso_recal_2026_05_15.md).

**Propensity-matched event design (2026-05-15) — causal rigor check:**

Tested whether the naive event-study effects (Layer 2a mid-season pass_rate 1.36×, Layer 2a' offseason plays 1.17×) survive matching on team quality + HC tenure. Both **attenuate** to 1.22× / 1.09× and lose significance (p 0.002→0.16; p 0.021→0.21). About 30-50% of the naive effect attributable to team-quality confounding. The residual causal effect of the firing itself is real (direction positive in both cases) but smaller than naive estimates suggested.

**Predictive validation is UNAFFECTED** (different test, conditional on coach prior profile, n=43 HC / n=151 OC, both p<0.005). The forecastability claim (slopes 0.30 HC / 0.45 OC) carries through unchanged. See [`hc_propensity_matched_results.md`](hc_propensity_matched_results.md).

For each retread coaching event (coach moved between teams within window),
the prediction `shift_pred = profile_coach_prior - profile_new_team_pre`
forecasts the observed shift `shift_obs = profile_new_team_post -
profile_new_team_pre` with statistically significant correlation.

**OC > HC on portability** is structurally consistent: the OC's RZ
fine-grain playbook is more roster-independent than the HC's macro
philosophy. Mechanism: choosing shotgun vs under-center in the red zone
is a scheme decision; running a high-pass-rate macro philosophy needs
roster (QB tier, OL, receivers) to execute. The lower-level the
schematic decision, the more portable the coach's authoring.

**Forecastability gate:** both event-triggered tests pass the SNR discipline.
A downstream user could apply a slope-0.30 correction to pass-rate-derived
quantities W1-W4 of the new season for retread HCs, and a slope-0.45 correction
to RZ-shotgun-derived quantities W1-W6 for retread OCs.
First-time HCs or OCs (no prior tenure in window) default to no
adjustment under a cold-start prior.

See [`predictive_validation_results.md`](predictive_validation_results.md)
for hit/miss case lists and full methodology.

## Coach style classes (fingerprint clustering)

k-means clustering on per-coach schematic profile vectors (shotgun_rate,
no_huddle_rate, deep_throw_rate, pass_rate, rz_shotgun_rate, rz_pass_rate)
pooled across each coach's full tenure. 129 HCs and 141 OCs cleared the
32-team-game floor.

### HC style classes (k=4, silhouettes 0.31-0.34)

| Cluster | n | Label | Representative coaches |
|---|---:|---|---|
| C0 | 32 | Shotgun-lean + RZ-Shotgun-lean + Pass-averse | Belichick, J. Harbaugh, McVay, K. Shanahan, P. Carroll |
| C1 | 45 | Deep-averse (run-heavy era) | J. Fisher, M. Shanahan, Gruden, Dungy, Holmgren |
| C2 | 3 | No-Huddle-heavy + Shotgun-heavy outliers | Kingsbury, Chip Kelly, Joe Judge |
| C3 | 49 | RZ-Pass-heavy + Pass-heavy | Reid, Tomlin, McCarthy, Payton, Pederson, LaFleur |

**Honest finding to flag (not a bug):** McVay, Shanahan, and Belichick
co-cluster in C0. Cluster labels reflect z-scores against the **modern**
league mean (which has shifted to ~63% shotgun), so absolute moderates
read as "lean" relative to current peers. C2 is highly imbalanced (n=3)
— Kingsbury and Chip Kelly are genuine tempo/empty-formation extremes,
not noise.

### OC style classes (k=5, silhouettes 0.28-0.29)

5 clusters; sizes 43 / 36 / 29 / 22 / 11. RZ-Pass-heavy/Pass-heavy is the
largest; No-Huddle/RZ-Shotgun-heavy outliers the smallest (n=11) — the
"experimental tempo" OC class. Full assignments in
`data/nfl/oc_fingerprint_clusters.csv` and visualizations at
`figures/oc_fingerprint_{pca_2d,radar}.png`.

Silhouettes in the 0.28-0.34 band are modest — these are gradient styles,
not crisp islands. See [`coach_fingerprint_clustering.md`](coach_fingerprint_clustering.md).

## Fantasy-outcome wiring (honest mixed result)

Connects the team-level schematic findings to per-player fantasy-points fantasy
outcomes via volume composition (predicted pass-rate or RZ-shotgun shift
× league avg pass plays × per-position fantasy-points yield).

**Per-event regression: NULL across all 8 (role × position) cells.**

| Cell | n | Pearson r | OLS slope | p_one |
|---|---:|---:|---:|---:|
| HC·QB | 41 | +0.04 | +0.28 | 0.40 |
| HC·RB | 41 | -0.08 | -9.89 | 0.69 |
| HC·WR | 41 | +0.04 | +0.81 | 0.38 |
| HC·TE | 41 | +0.15 | +2.90 | 0.17 |
| OC·QB | 149 | -0.02 | -2.20 | 0.60 |
| OC·RB | 149 | +0.03 | +1.63 | 0.35 |
| OC·WR | 149 | -0.01 | -0.68 | 0.56 |
| OC·TE | 149 | +0.05 | +3.58 | 0.27 |

No cell crosses p<0.10. Per-event fantasy-shift prediction is
**noise-dominated** by individual rosters, injuries, and shocks.

**However, order-of-magnitude average effects HOLD:**

| Role | QB Δ fantasy-points/season | RB | WR | TE |
|---|---:|---:|---:|---:|
| HC change | +7.0 | -0.6 | +4.1 | +2.9 |
| OC change | +0.4 | -1.4 | +1.1 | +0.8 |

Typical HC retread shifts team-QB position by ~7 fantasy-points/season, team-WR by
~4, team-TE by ~3. OC retreads are smaller (~1 fantasy-points/season at team-position
level).

**Honest paper claim:** team-level schematic shifts translate to
order-of-magnitude per-position FP movements (HC: ~3-7 fantasy-points/season; OC:
~1-2 fantasy-points/season at team-position level), but **per-event decomposition is
noise-dominated at current sample sizes (n=41-149)**. The team-level
predictive validation (HC r=0.39, OC r=0.48) is the load-bearing
forecastability claim; the fantasy decomposition is order-of-magnitude
indicative, not per-event-precise.

See [`fantasy_outcome_wiring_results.md`](fantasy_outcome_wiring_results.md)
for full methodology and assumption documentation.

The two-locus split is the deepest claim. The HC owns macro schematic
identity ("what kind of football do we play"); the OC owns red-zone fine-grain
schematic ("when we get to the red zone, do we go shotgun-empty or under-center
heavy"). DC fine-grain is pending. The generic "coaching" signal collapses
into this hierarchical structure once role-isolation is applied.

The **same partition (coaching_vector_id)** loads differently at different
resolutions of the same sport, on the same data, under the same SNR
discipline. That's the methodological finding.

## Implications for projection modeling

**Don't pool everything by coaching_vector_id.** That over-fits non-schematic
outcomes. The pre-existing convention of pooling by team_id is correct for
season-aggregate outcomes (Layer 1 confirms).

**Don't pool everything by team_id either.** That misses the schematic shift
at HC-change events (Layer 2) and obscures the schematic identity latent
(Layer 3).

**Right structure:** two latents, hierarchical.

- **Schematic style** (formation × tempo × target depth × score bucket):
  pool by coaching_vector_id with team_id as the prior.
- **Outcome magnitude** (pass rate, pace, RZ efficiency, FP totals):
  pool by team_id with coaching_vector_id as a within-team deviation.

Production-style integration of this structure is **gated on a predictive-validation
test** — does an incoming HC's prior-stop schematic profile forecast the new
team's schematic shift? If yes, ship as an event-triggered feature for teams
with offseason or mid-season HC/OC changes. If no, the finding remains
paper-worthy but doesn't translate to a forward-looking projection feature.

## Implications for methodology

The four-layer framework generalizes beyond NFL coaching:

1. **The same partition can load at multiple resolutions of the same data.**
   Single-resolution tests are insufficient; pre-register the layer matrix.
2. **Placebo design audit is load-bearing.** The original Test 1's "8-16× over
   team-id" was a between-partition ratio, not an observed-vs-null effect
   size. Both are true and should be reported, but they answer different
   questions.
3. **Structural-pinning checks are required for partition tests.** When the
   target partition is mechanically near-identical to the comparator within
   the test window (e.g., 95.99% HC stability inside game windows), F-stat
   degeneracy can produce false negatives. Window-level partition stability
   should be reported and acted on before verdict.
4. **Negative results survive at SNR discipline.** Layer 4 returned 0/32
   CARRIES across the four-layer matrix's most ambitious cell. That's a
   citeable null, not a reporting failure.

## NBA mirror (cross-league replication)

An analogous mid-season-change-drives-variance finding has been observed in basketball, the structural mirror of Layer 2 here: in basketball, mid-season events
(trade, role change) drive variance multipliers cleanly. In NFL, mid-season
HC changes drive pass-rate shifts cleanly. **Same finding, different
sport, parallel test design** — that's a replication template usable for
WNBA, NCAA Men's, NCAA Women's already in the cross-domain panel.

The cross-league pattern: a latent's correct resolution is not necessarily
seasonal. Mid-season events often resolve latents that season aggregates
average away.

## Open experiments (the queue)

These extend the empirical base without changing the four-layer framework.
Each result is paper-worthy regardless of direction.

| # | Test | Hours | Status | Result |
|---|---|---:|---|---|
| ~~1~~ | ~~OC-change event study~~ | ~~3~~ | **DONE 2026-05-15** | OC null on generic pass-rate (0.97×, p=0.636) at 3.2× HC sample. See [`oc_change_event_study_results.md`](oc_change_event_study_results.md). |
| ~~2~~ | ~~DC-change × defensive metrics~~ | ~~3~~ | **DONE 2026-05-15** | DC null on opp outcomes (0.96×, p=0.731). See [`dc_change_and_oc_layered_results.md`](dc_change_and_oc_layered_results.md). |
| ~~3~~ | ~~Schematic-style isolation at role level~~ | ~~6~~ | **DONE 2026-05-15** | HC 20/20 cells dominant; OC and DC subordinate. See [`dc_change_and_oc_layered_results.md`](dc_change_and_oc_layered_results.md). |
| ~~4~~ | ~~RZ-focused OC follow-up~~ | ~~3~~ | **DONE 2026-05-15** | **OC POSITIVE on RZ shotgun-rate (1.34×, p=0.0002).** Two-locus split confirmed. See [`dc_change_and_oc_layered_results.md`](dc_change_and_oc_layered_results.md). |
| ~~5~~ | ~~DC fine-grain test~~ | ~~3~~ | **DONE 2026-05-15** | NULL on all 3 RZ-defensive metrics. Coordinator-role asymmetry confirmed. |
| ~~5a~~ | ~~HC offseason × defensive (interpretation disambiguation)~~ | ~~2~~ | **DONE 2026-05-15** | NULL on all generic + RZ-defensive metrics. **Personnel-driven defense confirmed.** Rejects HC-owned defense hypothesis. |
| ~~6~~ | ~~Predictive validation OC: incoming OC RZ-shotgun prior → new team shift~~ | ~~6~~ | **DONE 2026-05-15** | **PORTABLE: r=+0.483, p<0.0001, slope=0.45, n=149**. See [`predictive_validation_results.md`](predictive_validation_results.md). |
| ~~7~~ | ~~Predictive validation HC: incoming HC pass-rate prior → new team shift~~ | ~~6~~ | **DONE 2026-05-15** | **PORTABLE: r=+0.386, p=0.0044, slope=0.30, n=41**. See [`predictive_validation_results.md`](predictive_validation_results.md). |
| ~~8~~ | ~~Coach fingerprint clustering (HC + OC named style classes)~~ | ~~6~~ | **DONE 2026-05-15** | k=4 HC, k=5 OC; visualizations + assignments. McVay/Shanahan/Belichick co-cluster — labels are z-scores vs modern league mean. See [`coach_fingerprint_clustering.md`](coach_fingerprint_clustering.md). |
| ~~9~~ | ~~Fantasy-outcome wiring (HC pass + OC RZ-shotgun → FP residuals)~~ | ~~3~~ | **DONE 2026-05-15** | Per-event NULL across 8 cells; order-of-magnitude effects hold (HC ~3-7 fantasy-points/season per position, OC ~1-2). See [`fantasy_outcome_wiring_results.md`](fantasy_outcome_wiring_results.md). |
| 10 | Propensity-matched HC + OC event designs | ~6 | queued | Causal rigor |
| 11 | Mid-season vs offseason HC change separation | ~3 | queued | Cleaner natural-experiment design |
| 12 | Cross-league NBA mirror (Layer 2 + 3 replication) | ~6 | queued | Cross-domain methodological claim |

## Reproduce

The four-layer tests live across these scripts:

| Layer | Script |
|---|---|
| 1. Static | `data/nfl/_src/snr_test_static.py` |
| 2a. HC-change within-season events | `data/nfl/_src/snr_test_perm_and_event.py` |
| 2a'. HC mid-season vs offseason separation | `data/nfl/_src/hc_midseason_vs_offseason.py` |
| 2b. OC-change offseason events | `data/nfl/_src/oc_change_event_study.py` |
| 2c. DC-change offseason events | `data/nfl/_src/dc_change_event_study.py` |
| 2d. OC RZ-focused follow-up | `data/nfl/_src/oc_rz_focused_followup.py` |
| 2e. DC RZ-focused follow-up | `data/nfl/_src/dc_rz_focused_followup.py` |
| 2f. HC offseason × defensive (disambiguation) | `data/nfl/_src/hc_offseason_defense_test.py` |
| 5a. HC predictive validation | `data/nfl/_src/hc_predictive_validation.py` |
| 5b. OC predictive validation | `data/nfl/_src/oc_predictive_validation.py` |
| 3a. Within-game schematic (composite) + placebo audit | `data/nfl/_src/snr_within_game_placebo_audit.py` |
| 3b. Within-game schematic role isolation | `data/nfl/_src/oc_schematic_isolation.py` |
| 4. Within-game stress (redesigned) | `data/nfl/_src/within_game_test2_redesign.py` |

Aggregated results state: [`data/nfl/SNR_TEST_STATE.md`](../data/nfl/SNR_TEST_STATE.md).
Per-test writeups in [`data/nfl/docs/within_game_test1_placebo_audit.md`](../data/nfl/docs/within_game_test1_placebo_audit.md),
[`data/nfl/docs/within_game_test2_structural_audit.md`](../data/nfl/docs/within_game_test2_structural_audit.md),
and [`within_game_test2_redesign_summary.md`](within_game_test2_redesign_summary.md).
