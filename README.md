# two-locus-nfl-coaching

Replication package for **"The Coach Locus is Offensive and Two-Tiered: A Multi-Resolution Event-Study Decomposition of NFL Coaching Effects with a Bounded Defensive Null"** (Humphrey, 2026).

Paper PDF: [docs/Two_Locus_NFL.pdf](docs/Two_Locus_NFL.pdf)

## What's here

A four-layer decomposition of NFL coaching effects tested at four resolutions of the same partition (`(HC, OC, DC)` triple at each team-season) under identical SNR discipline:

1. **Layer 1** — season-aggregate static test
2. **Layer 2** — within-season event study around coaching changes (with propensity matching)
3. **Layer 3** — within-game schematic identity by score bucket
4. **Layer 4** — within-game stress adaptation (0/32 NULL, reported)
5. **Layer 5** — predictive validation (forecastability of incoming coach prior profile)

Plus:
- Coach fingerprint clustering (k=4 HC, k=5 OC, silhouettes 0.28–0.34)
- Defensive personnel vs DC variance decomposition under permutation-null correction
- Continuous-staff placebo + Reid-PHI-2013 influence robustness checks
- Pre-registered teammate-composition signal-check (rejected at registered thresholds)

## Headline findings

| Layer | Finding | Magnitude |
|---|---|---|
| 2a (HC mid-season) | pass-rate shift | 1.36× p=0.002 (n=45) |
| 2a' (HC off-season) | plays/game shift | 1.17× p=0.021 (n=142) |
| 2d (OC off-season) | RZ shotgun-rate shift | 1.34× p=0.0002 (n=159) |
| 5 (HC predictive) | r=0.39, p=0.004, slope contaminated | existence-level only |
| 5 (OC predictive) | r=0.48, p<0.0001, slope=0.45 | placebo-confirmed |
| Defensive acute | 12 cells, MDE 1.16–1.22× at 80% power | bounded null |
| Defensive chronic | DC identity NET R² 0.14–0.18 on 3/7 cells | moderate signal |

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`
- ~5GB disk for data (downloaded by fetcher, not bundled)
- Internet for first-run data fetch

## Quick start

```bash
git clone https://github.com/mrnathanhumphrey-droid/two-locus-nfl-coaching
cd two-locus-nfl-coaching
pip install -r requirements.txt
bash reproduce.sh
```

End-to-end runtime ~45 min (data fetch ~15 min on first run, analysis ~30 min). Caching makes subsequent runs ~10× faster.

## Per-stage reproduction

```bash
bash reproduce.sh fetch              # data only (~15 min)
bash reproduce.sh layer1             # static test
bash reproduce.sh layer2             # event studies
bash reproduce.sh layer3             # within-game schematic
bash reproduce.sh layer5             # predictive validation + robustness
bash reproduce.sh propensity         # propensity matching
bash reproduce.sh defensive_chronic  # §5.3 chronic defensive test
bash reproduce.sh fingerprint        # coach clustering
bash reproduce.sh svi                # SVI-analogue decomposition
```

## Data sources

All data is free and public:

- **nflverse** — play-by-play, weekly rosters, participation, draft picks ([nflverse-data releases](https://github.com/nflverse/nflverse-data/releases))
- **Pro Football Reference** — coaching history (scraped at 1 req/sec via [`scripts/_fetch_coaching.py`](scripts/_fetch_coaching.py))

No paid data feeds are used.

## Directory layout

```
two-locus-nfl-coaching/
├── README.md                 you are here
├── LICENSE                   MIT
├── CITATION.cff              citation metadata
├── requirements.txt          Python deps
├── reproduce.sh              end-to-end orchestrator
│
├── scripts/                  18 analysis + 1 fetcher
│   ├── 00_fetch_data.py      download nflverse + PFR
│   ├── _fetch_coaching.py    PFR scraper (1 req/sec, idempotent)
│   ├── phase4_coach_pages.py PFR coach-page fetcher
│   │
│   ├── snr_test_static.py            Layer 1
│   ├── hc_midseason_vs_offseason.py  Layer 2a (HC events)
│   ├── oc_change_event_study.py      Layer 2b (OC events)
│   ├── oc_rz_focused_followup.py     Layer 2d (OC RZ-fine-grain)
│   ├── oc_schematic_isolation.py     Layer 3 (within-game F-stats)
│   ├── dc_change_event_study.py      Layer 2c (DC events)
│   ├── dc_rz_focused_followup.py     Layer 2e (DC RZ-fine-grain)
│   ├── hc_offseason_defense_test.py  Layer 2f (HC × defense)
│   ├── snr_test_within_game.py       Layer 3+4 (schematic + stress)
│   │
│   ├── hc_predictive_validation.py   Layer 5 — HC pass-rate portability
│   ├── oc_predictive_validation.py   Layer 5 — OC RZ-shotgun portability
│   ├── predictive_validation_robustness.py  placebo + Reid-influence
│   │
│   ├── hc_propensity_matched.py      causal rigor on Layer 2a
│   ├── defensive_personnel_vs_coaching_decomp.py  §5.3 chronic
│   ├── coach_fingerprint_clustering.py            k-means in style space
│   ├── snr_test_svi_analogue.py      team-id vs coaching-history
│
├── docs/                     paper + synthesis docs
│   ├── Two_Locus_NFL.pdf     the paper itself
│   ├── coaching_signal_findings.md   primary synthesis (32K)
│   ├── coach_fingerprint_clustering.md
│   ├── hc_midseason_vs_offseason_results.md
│   ├── oc_change_event_study_results.md
│   ├── dc_change_and_oc_layered_results.md
│   ├── hc_propensity_matched_results.md
│   ├── predictive_validation_results.md
│   ├── loso_recal_2026_05_15.md
│   ├── within_game_test1_placebo_audit.md
│   ├── within_game_test2_structural_audit.md
│   ├── within_game_test2_redesign_summary.md
│   ├── defensive_personnel_vs_coaching_summary.md
│   └── TEAMMATE_COMPOSITION_SIGNAL_CHECK_PREREG.md  pre-registration
│
├── figures/                  PCA + radar plots (HC k=4, OC k=5)
├── data/                     populated by fetcher (not committed)
│   ├── raw/                  nflverse + PFR HTML cache
│   └── processed/            built panels (state_residuals, staff, RZ)
└── results/                  analysis output CSVs
```

## Pre-registration

One pre-registered hypothesis was locked before data analysis: per-snap teammate-composition signal-check for three skill-position players (Kelce, Wilson, Pacheco). See [docs/TEAMMATE_COMPOSITION_SIGNAL_CHECK_PREREG.md](docs/TEAMMATE_COMPOSITION_SIGNAL_CHECK_PREREG.md) — verdict was NULL at registered thresholds and is reported in the paper's null catalog.

The four-layer decomposition itself was constructed iteratively under standing SNR discipline as each layer's result motivated the next. The paper §2.2 is explicit about this: "discipline-pre-registered, architecture-iterative."

## Citation

```bibtex
@article{humphrey2026coach,
  title   = {The Coach Locus is Offensive and Two-Tiered: A Multi-Resolution Event-Study Decomposition of NFL Coaching Effects with a Bounded Defensive Null},
  author  = {Humphrey, Nathan},
  year    = {2026},
  month   = {5}
}
```

See [CITATION.cff](CITATION.cff) for structured metadata.

## License

MIT. See [LICENSE](LICENSE).

## Contact

Issues / PRs welcome via GitHub. The paper PDF in [docs/](docs/) is the canonical source for the empirical claims; this code package is for replication.
