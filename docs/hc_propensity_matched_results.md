# Propensity-Matched HC Event Design — Causal Rigor Check

**Date:** 2026-05-15
**Status:** Honest finding. Both naive event-study effects (mid-season pass_rate 1.36×, offseason plays 1.17×) attenuate under propensity matching to 1.22× / 1.09× respectively, and neither reaches p<0.05 in the matched comparison. Predictive validation (the load-bearing forecastability claim) is **unaffected** and remains the load-bearing estimator.

## Why this test exists

Layer 2a (HC change → pass_rate shift 1.36×, p=0.002) and Layer 2a' (offseason HC change → plays/game shift 1.17×, p=0.021) compared firing-team-seasons to placebo team-seasons. HC firings are not random — they correlate with team quality (bad teams fire more) and within-season volatility. Anonymous referee concern, addressable directly: control for team-quality selection via propensity matching, see whether the event effect survives.

## Design

For each test:

1. **Build propensity model:** logistic regression of treatment label on pre-period team features
2. **Match treated to control 1:1 nearest-neighbor** within caliper 0.05 propensity SD
3. **Compute |Δ metric| within each pair** using identical windowing logic
4. **Compare matched-pair distributions** via 5000-perm null on mean difference

### Propensity covariates

**Mid-season pass_rate test:**
- `prior_wins` (wins in season N-1)
- `prior_pass_rate_residual_mean` (avg residual in season N-1)
- `hc_tenure_years` (consecutive prior same-HC seasons; OK here because tenure doesn't deterministically predict mid-season firing)
- `season` (year, for league trends)

**Offseason plays test:**
- `prior_wins`
- `prior_plays_residual_mean`
- `season`
- (hc_tenure_years EXCLUDED — treated team-seasons have rookie HC tenure=1 by definition; including it causes perfect separation with zero overlap)

## Results

| Test | Naive (placebo) | Propensity-matched | Δ ratio | Δ p |
|---|---|---|---|---|
| Mid-season pass_rate | 1.36× (p=0.002, n_real=45) | **1.22× (p=0.16, n=40 pairs)** | −0.14 | +0.16 |
| Offseason plays | 1.17× (p=0.021, n_real=142) | **1.09× (p=0.21, n=102 pairs)** | −0.08 | +0.19 |

Both effects attenuate by roughly 30–50%. Both remain directionally positive (treated |Δ| > control |Δ|). Neither reaches p<0.05 under the matched design.

Wider calipers (0.10, 0.20) tested — no rescue: at caliper 0.20, offseason gives ratio 1.11, p=0.14. Mid-season is sample-size-bound at 40 (only 40 firings have all covariates).

## Interpretation

About **30–50% of the naive effect was confounded by team-quality selection**:
- Bad teams have more within-season pass-rate variance regardless of coaching change → inflated mid-season placebo comparison
- Teams with HC changes have systematically different prior-year wins → inflated offseason placebo comparison

The **residual 22% / 9%** is consistent with a real but smaller causal effect of the firing itself. Power is limited (n=40 / 102 matched pairs) — the effect is directionally positive in both cases but not statistically significant in matched comparison.

## What this DOESN'T change

The **predictive validation finding is unaffected**. Predictive validation tests a DIFFERENT prediction: forecastability conditional on the incoming coach's prior profile, on a sample of *retread* coaches (n=43 HC, n=151 OC). It asks "does coach prior profile predict the direction of shift?" not "is the average event-effect non-zero?"

| Test | n | r | p | OLS slope | Verdict |
|---|---:|---:|---:|---:|---|
| HC pass-rate portability | 43 | +0.386 | 0.004 | 0.30 | PORTABLE |
| OC RZ-shotgun portability | 151 | +0.483 | <0.0001 | 0.45 | STRONGLY PORTABLE |

These p-values (0.004 and <0.0001) are far stronger than any of the event-study effects and would survive even substantial confounding adjustment. They also don't depend on the average event being non-zero — they depend on the *direction* of each event being predictable from coach-prior data.


## Honest paper framing

The Sloan submission should report this rigor check fully:

> "We tested whether the event-study effects (Layer 2a, 2a') survive propensity
> matching on team quality and HC tenure. Both effects attenuate by 30–50%
> and lose significance in the matched comparison (mid-season pass_rate
> 1.36× → 1.22×, p=0.002 → 0.16; offseason plays 1.17× → 1.09×,
> p=0.021 → 0.21). We conclude that part of the naive event-study effect
> is attributable to team-quality confounding, and the residual causal
> effect of the HC change itself is smaller than the naive estimate.
> Importantly, this attenuation does NOT affect the predictive
> validation finding (HC pass-rate portability r=+0.386 p=0.004 slope=0.30
> on n=43 retread events; OC RZ-shotgun portability r=+0.483 p<0.0001
> slope=0.45 on n=151). That test is conditional on incoming coach prior
> profile and remains the load-bearing forecastability claim."

## Reproduce

```bash
python scripts/hc_propensity_matched.py
```

Outputs:
- `results/hc_propensity_matched_results.csv` (2 rows)

## Limitations / future extensions

- **n=40 matched pairs for mid-season** is small. A wider HC-firing definition (include OC firings? interim regimes?) could power up but risks contaminating the natural experiment.
- **IPTW (inverse probability of treatment weighting)** as an alternative to strict matching — would use the full sample with weights. Likely modest gain since the propensity overlap region is narrow.
- **Doubly-robust estimator** (combining propensity weights with outcome regression) — closer to the canonical "causal effect of treatment" estimate, paper-worthy if we get to it.
- **Confound check for predictive validation:** symmetric design — does prior coach profile predict shift after matching on team quality? Expected to hold (the conditional signal is independent of the unconditional mean), worth verifying explicitly.
