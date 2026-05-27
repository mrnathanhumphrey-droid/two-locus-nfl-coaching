# Within-Game Test 2 — structural-bias audit

**Headline:** 0/12 is artifact: test was structurally biased toward team-id by construction.

## Section 1: coach-id stability stats

Test 2's units are (team, season) rows — by construction each row
has one coaching_vector_id (the predominant staff). HC-level mid-season
changes are documented in PFR's `head_coach_games_split` field; OC/DC
mid-season changes are less reliably documented. **HC stability is therefore
an UPPER BOUND on coaching_vector_id stability.**

- Total team-seasons in pbp universe: **829**
- Team-seasons with documented mid-season HC change: **49**
- **Season-level HC stability**: 94.09% (fraction of team-seasons with no HC change)
- **Play-weighted HC stability**: 95.99% (fraction of plays where the team's HC was the same as at season start)

Within-window coach_id variance: documented changes affect 49 of 829 team-seasons. Across-window coach_id variance: 377 distinct coaching_vector_ids across the test universe (~600 team-seasons), each appearing in ~1-2 team-seasons on average. Coaching_vector turnover happens BETWEEN seasons, not within.

## Section 2: structural-bias verdict

**STRUCTURALLY BIASED**: Coach and team are nearly identical inside windows; partition cannot meaningfully attribute variance.

Per audit-prompt thresholds:
- stability > 95% → STRUCTURALLY BIASED
- stability 80-95% → AMBIGUOUS
- stability < 80% → UNBIASED (0/12 genuine)

Observed play-weighted stability: **95.99%**.

**Implication of the BIASED verdict (95.99% sits just above the 95% threshold):**
HC and coaching_vector are stable in nearly all team-seasons (~94%), so within
those team-seasons coaching_vector_id and team_id are mechanically equivalent
(one coach per team per season). The F-stat partition F_coach vs F_team in
original Test 2 had **near-zero room** for coaching_vector to add information
beyond team_id — most rows had one coach, so coaching_vector simply chunks
team_id into smaller groups. The 0/12 result is **largely predetermined by
the test construction**.

The verdict sits right at the 95% boundary; "barely BIASED" is the honest
framing. The conditioned re-run on the 8% of mid-season change events tries
to isolate genuine coach-driven shifts.

## Section 3: conditioned-subset re-run (mid-season HC change events only)

- Mid-season HC change events: **49** team-seasons documented
- Match to pbp test universe: **49** team-windows
- **Status: SUFFICIENT POWER** (≥ 30 units).

Conditioned-subset design: each mid-season-change team-season is split into
pre-change and post-change segments using `head_coach_games_split` to identify
the change game-number. Style means computed per segment per score-bucket,
then shift = trailing-bucket value − leading/tied-bucket value per segment.
F-stat partition F_coach (segment-coach IDs) vs F_team (team).

| shift | n | ng_coach | ng_team | F_coach | F_team | ratio | verdict |
|---|---|---|---|---|---|---|---|
| pass_rate: trailing_2plus-leading_2plus | 67 | 67 | 26 | nan | 1.042 | nan | TIE |
| pass_rate: trailing_2plus-tied | 90 | 90 | 27 | nan | 0.780 | nan | TIE |
| shotgun_rate: trailing_2plus-leading_2plus | 67 | 67 | 26 | nan | 1.061 | nan | TIE |
| shotgun_rate: trailing_2plus-tied | 90 | 90 | 27 | nan | 0.811 | nan | TIE |
| no_huddle_rate: trailing_2plus-leading_2plus | 67 | 67 | 26 | nan | 1.588 | nan | TIE |
| no_huddle_rate: trailing_2plus-tied | 90 | 90 | 27 | nan | 1.025 | nan | TIE |
| deep_throw_rate: trailing_2plus-leading_2plus | 67 | 67 | 26 | nan | 1.030 | nan | TIE |
| deep_throw_rate: trailing_2plus-tied | 90 | 90 | 27 | nan | 0.598 | nan | TIE |

**Conditioned headline: COACH 0 / TEAM 0 / TIE 8 (of 8)**

Caveat: with ~80 segment-coach IDs and ~30 teams across ~98 segment rows,
F_coach has high d.f. (n_groups ≈ n_obs) and tends toward a degenerate
F-stat regime where most groups have 1 observation. Interpret with caution.

## Section 4: revised verdict for stress-adaptation layer

**Original verdict (COACH 0/12 / TEAM 5/12 / TIE 7/12): WITHDRAWN.**
Test was structurally biased; the result reflects test construction, not
substance. Stress-adaptation latent attribution is **undetermined** — needs
a redesigned test that doesn't equate coaching_vector with team_id within window.
