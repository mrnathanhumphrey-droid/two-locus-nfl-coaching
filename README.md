# two-locus-nfl-coaching

Replication package for **"The Portable Coach: Which NFL Coaching Effects Transfer Across Rosters, and How
Confidently"** (Humphrey, 2026; extended abstract: [docs/Sloan_Two_Locus_NFL_Abstract.pdf](docs/Sloan_Two_Locus_NFL_Abstract.pdf)).
The internal name of the analysis is *two-locus*.

**One command reproduces the layered analysis from frozen inputs** (no download, no scraping). The run ends by
checking each output against `results/expected/`. See [Scope](#scope) for which of the abstract's results this
package covers today.

```bash
git clone https://github.com/mrnathanhumphrey-droid/two-locus-nfl-coaching
cd two-locus-nfl-coaching
pip install -r requirements.txt          # pinned: the versions the expected results were produced with
bash reproduce.sh                        # about 5 minutes; ends with "submitted: 27 of 27 outputs match"
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

Headline values, as produced by `bash reproduce.sh` (file: `results/expected/submitted/`):

| Finding | Value | File |
|---|---|---|
| HC mid-season change, pass-rate shift | 1.364×, p=0.002, n=45 | `hc_midseason_vs_offseason_results.csv` |
| HC off-season change, plays shift | 1.169×, p=0.021, n=142 | `hc_midseason_vs_offseason_results.csv` |
| OC change, red-zone shotgun-rate shift | 1.343×, p=0.0002, n=175 | `oc_rz_focused_summary.csv` |
| HC portability (pass rate) | r=0.386, one-sided p=0.0044, slope 0.306, n=43 | `hc_predictive_validation_summary.csv` |
| OC portability (red-zone shotgun rate) | r=0.483, p<0.0001, slope 0.460, n=151 | `oc_predictive_validation_summary.csv` |

## Scope

This package reproduces the layered event-study analysis, the headline effects above, and the predictive
validation, which estimates its slopes by ordinary least squares.

Several results in the extended abstract come from later revision analyses that are **not yet in this package**:
- the hierarchical random-coefficient portability slopes (OC 0.469, HC 0.279, with 95% intervals)
- the concept-family portability slopes (0.41–0.50)
- the defensive-coordinator role-isolation result (opponent plays 6.37×)
- the NBA transfer check

They will be added under the same frozen-input, expected-output discipline. Until then, the OLS slopes here
(OC 0.460, HC 0.306) are the predictive-validation estimates this package reproduces.

## Two data windows

The study ran in two passes, and each test reproduces on the data it ran on:
- **Seasons through 2024:**
  - Layers 1 to 3
  - the event studies
  - within-game
  - the defensive decomposition
  - team identity
  - fingerprints
- **Through 2025:** the predictive validation, its robustness checks and propensity matching, which were re-run
  after the 2025 season was added.

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
data/frozen/          the inputs (see data/frozen/SOURCES.md), with MANIFEST.sha256
scripts/              the analysis scripts
scripts/build/        stage.py (data windows), panels.py + the original panel builders, check_results.py
results/expected/     the expected outputs, submitted and corrected
docs/                 the extended abstract (PDF) and the research notes behind each layer
figures/              fingerprint figures (regenerated by the run)
```

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

nflverse (CC-BY-4.0) and coaching facts compiled from Pro Football Reference, cited per row. Full attribution is
in [data/frozen/SOURCES.md](data/frozen/SOURCES.md).

## License

Code: MIT, see [LICENSE](LICENSE). Data: see [data/frozen/SOURCES.md](data/frozen/SOURCES.md).
