# two-locus-nfl-coaching

Replication package for **"The Portable Coach: Which NFL Coaching Effects Transfer Across Rosters, and How
Confidently"** (Humphrey, 2026; extended abstract: [docs/Sloan_Two_Locus_NFL_Abstract.pdf](docs/Sloan_Two_Locus_NFL_Abstract.pdf)).
The internal name of the analysis is *two-locus*.

**One command reproduces every number in the extended abstract from frozen inputs** (no download, no scraping).
The run prints each abstract figure next to the value it reproduced (see
[Every number in the abstract](#every-number-in-the-abstract)), then checks all 63 outputs against
`results/expected/`.

```bash
git clone https://github.com/mrnathanhumphrey-droid/two-locus-nfl-coaching
cd two-locus-nfl-coaching
pip install -r requirements.txt          # pinned: the versions the expected results were produced with
bash reproduce.sh                        # about 12 minutes; ends with "submitted: 63 of 63 outputs match"
```

| Command | What it does |
|---|---|
| `bash reproduce.sh` | Runs every test on the inputs the paper used. Writes `results/submitted/` and checks it against `results/expected/submitted/`. |
| `bash reproduce.sh verify` | Rebuilds the three analysis panels from the raw frozen files with the original builders, and checks they equal the frozen panels. |
| `bash reproduce.sh corrected` | Runs the same analysis with every team under one team-code standard (see [Team codes](#team-codes)). Checks against `results/expected/corrected/`. |

Requires Python 3.12 and bash. On Windows, Git Bash works.

## What the tests are

The `(HC, OC, DC)` staff of each team-season is tested at several resolutions under one permutation-null
discipline (real coaching changes vs continuous-staff placebo team-seasons).

| Layer | Script | Output |
|---|---|---|
| 1 Season-aggregate static | `snr_test_static.py` | `snr_test_static_*.csv` |
| 2a HC mid-season / off-season | `hc_midseason_vs_offseason.py` | `hc_midseason_vs_offseason_results.csv` |
| 2b OC changes | `oc_change_event_study.py` | `oc_change_event_*.csv` |
| 2c DC changes | `dc_change_event_study.py` | `dc_change_event_*.csv` |
| 2d OC red-zone detail | `oc_rz_focused_followup.py` | `oc_rz_focused_*.csv` |
| 2e DC red-zone detail | `dc_rz_focused_followup.py` | `dc_rz_focused_*.csv` |
| 2f HC changes × defense | `hc_offseason_defense_test.py` | `hc_offseason_defense_*.csv` |
| 3 Within-game schematic | `oc_schematic_isolation.py`, `snr_test_within_game.py` | `oc_schematic_isolation_results.csv`, `snr_within_game_*.csv` |
| 5 Predictive validation | `hc_predictive_validation.py`, `oc_predictive_validation.py` | `*_predictive_validation_*.csv` |
| 5 Placebo + influence checks | `predictive_validation_robustness.py` | `predictive_validation_robustness.csv` |
| Propensity matching | `hc_propensity_matched.py` | `hc_propensity_matched_results.csv` |
| Defensive chronic (personnel vs DC) | `defensive_personnel_vs_coaching_decomp.py` | `defensive_svi_decomposition.csv` |
| Team identity vs coaching history | `snr_test_svi_analogue.py` | `snr_svi_*.csv` |
| Coach fingerprints | `coach_fingerprint_clustering.py` | `*_fingerprint_clusters.csv`, `figures/` |

**The revision analyses** (`scripts/revision/`, outputs in `results/<mode>/revision/`) produce the abstract's
portability, concept-family and defensive-coordinator results:

| Analysis | Script | Output |
|---|---|---|
| Hierarchical random-coefficient portability (PyMC, 4 chains, seed 42) | `fit_rc_event_study_pymc.py` | `oc_rc_summary.json`, `hc_rc_summary.json`, posteriors |
| Portability with team-quality covariates | `rc_propensity_refit.py` | `*_propensity_refit.json` |
| Hierarchical slope vs the OLS placebo distribution (Figure 1) | `rc_placebo_and_propensity.py` | `placebo_comparison.json` |
| OC concept-family portability (Figure 2) | `oc_concept_family_portability.py`, `make_sloan_figures.py` | `oc_concept_family_results.csv`, `figures/` |
| DC role isolation, by era | `dc_within_hc_event_study.py`, `build_extended_defensive_panel.py`, `dc_within_hc_extended.py`, `dc_within_hc_era_strat.py` | `dc_within_hc_*.csv` |

**The NBA transfer check** (`scripts/nba/`, outputs in `results/<mode>/nba/`) runs the same framework on the NBA.
Opponent three-point rate is the defensive locus that survives the naive event study, roster-retention matching
(`propensity_with_roster_results.csv`) and predictive validation (`opp_3pt_predictive_validation_summary.csv`).

## Every number in the abstract

`bash reproduce.sh` prints this table from the run's own outputs (`scripts/build/check_abstract.py`):

| Abstract figure | Reproduced | File |
|---|---|---|
| HC mid-season firing, pass rate 1.36× (p=.002) | 1.3640, p=0.0020 | `hc_midseason_vs_offseason_results.csv` |
| HC off-season change, pace 1.17× (p=.021) | 1.1688, p=0.0210 | `hc_midseason_vs_offseason_results.csv` |
| OC change, red-zone shotgun rate 1.34× (p=.0002) | 1.3425, p=0.0002 | `oc_rz_focused_summary.csv` |
| OC portability slope 0.469 [0.32, 0.62] | 0.4690 [0.3202, 0.6208] | `revision/oc_rc_summary.json` |
| HC portability slope 0.279 [−0.01, 0.56] | 0.2790 [−0.0082, 0.5546] | `revision/hc_rc_summary.json` |
| Six concept families porting at slopes 0.41–0.50 | 6 families, 0.4105–0.4950 | `revision/oc_concept_family_results.csv` |
| DC change 2017–2024, opponent plays 6.37× (p=.007) | 6.3693, p=0.0070 | `revision/dc_within_hc_era_strat.csv` |
| NBA: head-coach perimeter defense survives | survives roster matching (1.2754×, p=0.0482) | `nba/propensity_with_roster_results.csv` |

The HC interval's upper end reproduces as 0.5546, which rounds to 0.55; the abstract prints 0.56.

## Two data windows

The study ran in two passes, and each test reproduces on the data it ran on:
- **Seasons through 2024:**
  - Layers 1 to 3
  - the event studies
  - within-game
  - team identity
  - fingerprints
- **Through 2025:**
  - the predictive validation, its robustness checks and propensity matching, re-run after the 2025 season was
    added
  - the revision analyses
  - the defensive decomposition, whose results are identical in either window; it runs here because the
    revision analyses read its defensive outcome panel

The NBA analysis reads its own frozen inputs and has no window.

`reproduce.sh` stages each window in turn (`scripts/build/stage.py`).

## Team codes

The play-by-play uses each franchise's current code. The schedule, roster and injury files use the code of the
time (STL, SD and OAK; the rosters also use SL, ARZ, BLT, CLV and HST). In the submitted analysis, joins across
those files did not match the following, so those team-games carry no residual and drop out of the tests:
- the St. Louis Rams 2000–2015
- the San Diego Chargers 2000–2016
- the Oakland Raiders 2000–2019
- the roster-continuity seasons in which a code changed

`bash reproduce.sh` reproduces the submitted analysis exactly as computed. `bash reproduce.sh corrected` applies
one code standard before every join (`CANON` in `scripts/build/panels.py`) and runs the same, unchanged scripts.
Its outputs are in `results/expected/corrected/`.

## Layout

```
data/frozen/          the inputs (see data/frozen/SOURCES.md), with MANIFEST.sha256; NBA inputs in data/frozen/nba/
scripts/              the layered analysis
scripts/revision/     the revision analyses
scripts/nba/          the NBA transfer check
scripts/build/        stage.py (data windows), panels.py + the original panel builders, check_results.py,
                      check_abstract.py
results/expected/     the expected outputs, submitted and corrected (63 each)
docs/                 the extended abstract (PDF) and the research notes behind each layer
figures/              fingerprint and abstract figures (regenerated by the run)
```

The analysis scripts are the research code with only their path lines changed. Three scripts that printed a
result without saving it now also write it to a CSV:
- `dc_within_hc_era_strat.py`
- `propensity_with_roster.py`
- `nextgen_propensity.py`

`snr_within_game_placebo_audit.py`, a helper module `oc_schematic_isolation.py` imports, was missing from the
first release and is restored from the research code.

`scripts/00_fetch_data.py`, `_fetch_coaching.py` and `phase4_coach_pages.py` are the original collection scripts,
kept for provenance. They are not needed to reproduce the results. The live sources have changed since the data
was collected, so they will not rebuild the frozen inputs byte for byte. The Pro Football Reference scripts wait
4 seconds between requests and need `curl_cffi` plus a Cloudflare clearance cookie from your own browser
(`data/raw/pfr/_session.py`, not included).

## Pre-registration

One hypothesis was pre-registered before the data was touched: the per-snap teammate-composition signal check.
See `docs/TEAMMATE_COMPOSITION_SIGNAL_CHECK_PREREG.md`. Its verdict was null at the registered thresholds. The
layered decomposition itself was built iteratively, each layer's result motivating the next.

## Data

Every dataset is included exactly as the analysis used it, attributed to its source:
- nflverse (CC-BY-4.0)
- NFL coaching facts compiled from Pro Football Reference
- NBA box scores and tracking from nba.com
- NBA coaching facts compiled from Basketball-Reference

Full attribution is in [data/frozen/SOURCES.md](data/frozen/SOURCES.md).

## License

Code: MIT, see [LICENSE](LICENSE). Data: see [data/frozen/SOURCES.md](data/frozen/SOURCES.md).
