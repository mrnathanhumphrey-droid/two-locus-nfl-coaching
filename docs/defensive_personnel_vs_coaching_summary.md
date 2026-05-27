# Defensive Personnel vs Coaching — Variance Decomposition Probe

**Date:** 2026-05-22
**Probe:** SVI-analogue decomposition of defensive outcome residuals across (a) defensive personnel composition index (prior- and current-season snap-weighted draft capital) and (b) DC identity partition (single-block) and (HC, DC) coaching_vector_id (two-block).
**Script:** `data/nfl/_src/defensive_personnel_vs_coaching_decomp.py`
**Outputs:**
- `data/nfl/defensive_personnel_index.parquet` (288 team-seasons 2016-2024)
- `data/nfl/defensive_outcome_panel.parquet` (288 team-seasons, 7 cells)
- `data/nfl/defensive_personnel_vs_coaching_joined.parquet` (253 team-seasons after coach join)
- `data/nfl/defensive_svi_decomposition.csv` (14 rows: 7 cells × 2 partitions)

## Headline

**The personnel-dominates-defense hypothesis is NOT supported by this probe.** Personnel index (snap-weighted draft capital) adds zero unique variance beyond DC identity. DC identity partition contains real signal at 3-5 of 7 defensive cells under permutation-null correction — sharpens the §5.1 defensive null from "DC change events show no detectable shift" to "DC change events show no detectable shift AND a snap-weighted draft-capital personnel index adds no marginal predictive content beyond DC identity for any defensive cell tested."

**This is a real result, not a strengthening of the asymmetric authoring frame.** Reporting honestly per pre-existing methodological discipline (audit-driven exploratory falsification).

---

## Methodology

### Personnel index construction

Per (team, season) for 2016-2024:
1. Snap counts per defensive player from `pbp_participation_*.parquet` (9 seasons)
2. Position joined from `rosters/weekly_rosters_*.parquet` → filter to defensive positions
3. Draft pick joined from `combine/draft_picks.parquet`; converted to draft value via Jimmy-Johnson-style decay: `value(pick) = 3000 / (1 + 0.3(pick-1))^1.4`; undrafted = 0
4. Three indices computed per (team, season):
   - `personnel_index` — snap-weighted draft value across all defensive players
   - `top11_mean_draft_value` — mean draft value of 11 most-snapped defenders
   - `top22_mean_draft_value` — mean draft value of 22 most-snapped defenders

### Defensive outcome cells

Per (team, season), 7 defensive cells:
- `opp_pass_rate_residual` (from offensive `state_residuals.parquet` inverted to defending team)
- `opp_plays_residual`
- `opp_rz_trips_residual`
- `opp_epa_per_play`
- `opp_rz_pass_rate_residual` (PBP-derived; league-mean residual)
- `opp_rz_sack_rate_residual`
- `opp_rz_td_rate_residual`

### Decomposition

For each cell, fit three OLS models:
- **Block A:** outcome ~ personnel_index + top11_mean_draft_value + top22_mean_draft_value
- **Block B:** outcome ~ DC_dummies (DC_only) OR coaching_vector_id_dummies (HC+DC)
- **Block A+B:** outcome ~ personnel features + coaching dummies (both)

Then compute:
- `unique(personnel | coaching) = R²(A+B) - R²(B)`
- `unique(coaching | personnel) = R²(A+B) - R²(A)`
- `overlap = R²(A) + R²(B) - R²(A+B)`

### Permutation-null correction (load-bearing for honest interpretation)

Raw R² for the coaching block is inflated by high-dimensional categorical overfitting (93 DC dummies, 133 coaching_vector dummies, n=253). Per the same discipline applied in the existing SVI-analogue work, computed null distribution via N=200 permutations of the coaching label assignment. Reported NET values subtract the per-permutation mean: `R²_NET = R²_observed - R²_null_mean`.

The personnel block (3 features) has negligible null overfit and is left uncorrected.

---

## Results — RAW R²

Cohort: n=253 team-seasons spanning 2017-2024 (intersection of personnel index + outcome panel + coach roster). 94 unique DCs, 134 unique coaching_vector_ids.

### DC-only partition

| Cell | R²_personnel | R²_DC | R²_both | unique_pers | unique_DC | overlap |
|---|---:|---:|---:|---:|---:|---:|
| opp_pass_rate_residual | 0.040 | 0.525 | 0.544 | 0.019 | 0.504 | 0.021 |
| opp_plays_residual | 0.029 | 0.512 | 0.529 | 0.017 | 0.499 | 0.012 |
| opp_rz_trips_residual | 0.018 | 0.311 | 0.330 | 0.018 | 0.312 | -0.001 |
| opp_epa_per_play | 0.062 | 0.408 | 0.443 | 0.036 | 0.381 | 0.026 |
| opp_rz_pass_rate_residual | 0.013 | 0.506 | 0.512 | 0.007 | 0.499 | 0.007 |
| opp_rz_sack_rate_residual | 0.012 | 0.393 | 0.402 | 0.009 | 0.390 | 0.003 |
| opp_rz_td_rate_residual | 0.024 | 0.391 | 0.415 | 0.024 | 0.391 | -0.000 |

### HC+DC coaching_vector partition

| Cell | R²_personnel | R²_CV | R²_both | unique_pers | unique_CV | overlap |
|---|---:|---:|---:|---:|---:|---:|
| opp_pass_rate_residual | 0.040 | 0.639 | 0.660 | 0.021 | 0.619 | 0.019 |
| opp_plays_residual | 0.029 | 0.705 | 0.715 | 0.009 | 0.685 | 0.020 |
| opp_rz_trips_residual | 0.018 | 0.511 | 0.559 | 0.048 | 0.541 | -0.030 |
| opp_epa_per_play | 0.062 | 0.620 | 0.674 | 0.054 | 0.612 | 0.008 |
| opp_rz_pass_rate_residual | 0.013 | 0.651 | 0.661 | 0.011 | 0.648 | 0.003 |
| opp_rz_sack_rate_residual | 0.012 | 0.490 | 0.510 | 0.020 | 0.498 | -0.007 |
| opp_rz_td_rate_residual | 0.024 | 0.616 | 0.628 | 0.012 | 0.604 | 0.012 |

**Surface read of raw R² says "DC dominates personnel 0/7." But raw R² is contaminated by the 93-dummy DC block overfit. The null-permutation mean R² for DC dummies is ~0.37 — meaning a randomly-shuffled DC label still yields R² of 0.37 on these outcomes from sheer high-dim categorical fit. Reporting raw R² unadjusted would be misrepresenting the test.**

---

## Results — PERMUTATION-NULL-CORRECTED

### DC-only partition (NET R² = R² - null mean)

| Cell | R²_pers | R²_DC_raw | null_mean | R²_DC_NET | uniq_pers_NET | uniq_DC_NET | overlap_NET | perm_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opp_pass_rate_residual | 0.040 | 0.525 | 0.372 | **+0.153** | -0.011 | **+0.102** | 0.051 | <0.005 |
| opp_plays_residual | 0.029 | 0.512 | 0.367 | **+0.145** | -0.007 | **+0.108** | 0.036 | <0.005 |
| opp_rz_trips_residual | 0.018 | 0.311 | 0.366 | -0.055 | +0.003 | -0.069 | 0.015 | NULL |
| opp_epa_per_play | 0.062 | 0.408 | 0.371 | +0.037 | -0.009 | -0.035 | 0.072 | NULL |
| opp_rz_pass_rate_residual | 0.013 | 0.506 | 0.364 | **+0.141** | -0.007 | **+0.121** | 0.020 | <0.005 |
| opp_rz_sack_rate_residual | 0.012 | 0.393 | 0.366 | +0.027 | -0.003 | +0.012 | 0.016 | NULL |
| opp_rz_td_rate_residual | 0.024 | 0.391 | 0.373 | +0.017 | +0.004 | -0.003 | 0.020 | NULL |

**DC partition is significant on 3/7 cells under perm null (α=0.05): opp_pass_rate, opp_plays, opp_rz_pass_rate.** All three are macro defensive tendencies (how often opponents pass, how many plays opponents run, how often opponents pass in the RZ). These are scheme-authored signals.

**DC partition is NULL on 4/7 cells:** opp_rz_trips, opp_epa_per_play, opp_rz_sack_rate, opp_rz_td_rate. RZ trips, EPA-per-play, and TD/sack rates are the OUTCOME cells §5.1 already flagged as DC-effects-not-detectable.

### HC+DC coaching_vector partition (NET R²)

| Cell | R²_pers | R²_CV_raw | null_mean | R²_CV_NET | uniq_pers_NET | uniq_CV_NET | overlap_NET | perm_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opp_pass_rate_residual | 0.040 | 0.639 | 0.523 | **+0.116** | -0.005 | **+0.071** | 0.045 | <0.005 |
| opp_plays_residual | 0.029 | 0.705 | 0.529 | **+0.176** | -0.010 | **+0.137** | 0.039 | <0.005 |
| opp_rz_trips_residual | 0.018 | 0.511 | 0.527 | -0.016 | +0.033 | -0.001 | -0.016 | NULL |
| opp_epa_per_play | 0.062 | 0.620 | 0.526 | **+0.094** | +0.019 | **+0.051** | 0.044 | <0.005 |
| opp_rz_pass_rate_residual | 0.013 | 0.651 | 0.526 | **+0.125** | -0.003 | **+0.109** | 0.016 | <0.005 |
| opp_rz_sack_rate_residual | 0.012 | 0.490 | 0.528 | -0.037 | +0.008 | -0.041 | 0.004 | NULL |
| opp_rz_td_rate_residual | 0.024 | 0.616 | 0.530 | **+0.086** | -0.008 | **+0.055** | 0.031 | <0.005 |

**HC+DC coaching_vector partition is significant on 5/7 cells under perm null:** opp_pass_rate, opp_plays, opp_epa_per_play, opp_rz_pass_rate, opp_rz_td_rate. Adding HC identity to DC identity recovers signal on opp_epa_per_play and opp_rz_td_rate that DC alone couldn't detect.

**Still NULL on 2/7:** opp_rz_trips, opp_rz_sack_rate.

---

## Interpretation

### What this finding IS

1. **The defensive null in §5.1 (DC change events vs placebo) is REFINED, not contradicted.** DC change events fire too sparsely to detect coaching effects, but DC IDENTITY held constant across a tenure does carry signal on 3-5 of 7 defensive cells. The "DC effects not detectable" claim is specific to the change-event design.

2. **Macro defensive tendencies (opp_pass_rate, opp_plays, opp_rz_pass_rate) are DC-authored** at moderate magnitude (NET R² 0.12-0.18 under perm null). These mirror the OFFENSIVE Layer 2a finding that HC mid-season changes shift pass_rate residual at 1.36×.

3. **RZ trips and sack-rate residuals are PERSONNEL- AND COACHING-NULL** — neither block has perm-significant signal. These cells genuinely look noise-dominated at the season-aggregate resolution.

4. **Personnel index adds no unique information beyond coaching identity.** unique(personnel | coaching) is at or near zero on every cell, in both partitions.

### What this finding IS NOT

1. **NOT "personnel dominates coaching."** Raw R² showed coaching dominating; NET R² shows coaching still dominating (or being null on shared cells). Personnel index ≠ dominant author on any cell.

2. **NOT a positive defensive-authoring claim that strengthens §5.1.** The brief proposed that this probe would convert the bounded null into a positive asymmetric finding. It did not. The honest result is "DC partition has real but modest signal on 3-5/7 cells; personnel proxy adds nothing marginal."

3. **NOT a final word on personnel.** The personnel index here is a single proxy (snap-weighted draft capital). Alternative indices may surface different results. Open candidates:
   - Prior-season individual defensive EPA contributions (richer talent signal, more endogenous to prior coaching)
   - Position-group-specific indices (CB quality vs LB quality vs DL quality as separate features)
   - PFF grades if available (would directly measure player-quality residual)
   - Snap-weighted prior-team prior-year EPA (most exogenous to current DC, hardest to compute)

The current draft-capital index is the most-exogenous-to-current-DC option but is also the noisiest measure of true defensive talent. A better proxy might recover personnel signal.

### Methodological lesson reinforced

Raw R² with high-dimensional categorical blocks is structurally biased toward the categorical block. The 93-DC-dummy partition has baseline-null R² of ~0.37 on these defensive outcomes — i.e., **a randomly-shuffled DC label assignment still produces R² = 0.37 from sheer overfit**. Reporting raw R² without null correction would have produced a spurious "coaching dominates defense" claim.

Same lesson the SNR_TEST_STATE.md walk-back applied to OFFENSIVE residuals (team-id vs coaching-history). Applies symmetrically to defensive residuals.

---

## What to write in §5

**Honest sharpening (replaces or augments existing §5.1 defensive null framing):**

> The defensive null surfaced in Layer 2a (DC change event vs placebo, n=144, all metrics null) refines under variance-decomposition analysis at the team-season identity resolution. Under permutation-null correction, DC identity contains detectable signal at 3 of 7 tested defensive cells (opp_pass_rate, opp_plays, opp_rz_pass_rate; perm-p < 0.05; NET R² 0.14-0.15). HC+DC coaching_vector_id partition recovers an additional 2 cells (opp_epa_per_play, opp_rz_td_rate; total 5 of 7).
>
> A snap-weighted draft-capital defensive personnel index adds zero unique variance beyond coaching identity across all 7 cells — the personnel-coaching joint variance is absorbed entirely into coaching identity at this index's granularity. Whether this reflects true absence of unique-personnel signal or weakness of the snap-weighted draft-capital proxy is open; alternative personnel indices (prior-year individual defensive EPA, position-group-stratified talent) remain unexplored at the time of this paper.
>
> The combination — DC change events null on all metrics (acute test) AND DC identity moderately significant on macro defensive tendencies (chronic test) — is consistent with slow-emerging defensive scheme authoring that escapes pre/post change-window detection but persists across tenure.

### What NOT to write

Don't claim "personnel dominates defense X:1." The data don't support it under proper null correction.

Don't claim "asymmetric authoring: offense=coaching, defense=personnel." The data show coaching authors both, just at different magnitudes (NET R² 0.15 defense vs higher on offensive Layer 2a 1.36× pass-rate event-study effect).

Don't claim "this strengthens the bounded null into a positive claim." The probe sharpens the null's domain (change events vs identity) without producing a complementary positive asymmetric finding.

---

## Files

| Artifact | Path |
|---|---|
| Analysis script | `data/nfl/_src/defensive_personnel_vs_coaching_decomp.py` |
| Personnel index parquet | `data/nfl/defensive_personnel_index.parquet` |
| Defensive outcome panel | `data/nfl/defensive_outcome_panel.parquet` |
| Joined panel for decomp | `data/nfl/defensive_personnel_vs_coaching_joined.parquet` |
| Decomposition table | `data/nfl/defensive_svi_decomposition.csv` |
| This summary | `docs/defensive_personnel_vs_coaching_summary.md` |

---

## Future work (if revisited)

1. **Position-stratified personnel index** — separate CB / LB / DL talent indices. Top defense quality is often concentrated at one position group (top-CB-heavy or top-DL-heavy rosters). Aggregating to single snap-weighted draft value loses this.
2. **Prior-year individual defensive EPA** — most direct measure of player quality, but partly endogenous to prior coaching (a player's prior-year EPA was authored partly by his prior DC). Use prior-team-different-DC residual as cleanest version.
3. **PFF defensive grades** if data acquisition path opens — directly measures player quality residual.
4. **Joint partition** — instead of separating personnel vs coaching, jointly model `outcome ~ personnel × coaching_interaction` and ask whether interaction terms beat additive. Identifies scheme-personnel match effects.
