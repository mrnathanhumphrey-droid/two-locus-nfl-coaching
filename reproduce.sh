#!/usr/bin/env bash
# End-to-end reproduction of the four-layer NFL coaching decomposition.
# Runs sequentially. ~30-45 min total once data is fetched (data fetch is
# ~10-15 min on first run, cached thereafter).
#
# Usage:
#   bash reproduce.sh         # full pipeline
#   bash reproduce.sh fetch   # data fetch only
#   bash reproduce.sh layer2  # layer 2 event studies only
#
# Outputs land in:
#   data/raw/        — nflverse + PFR caches
#   data/processed/  — built panels (state_residuals, staff, RZ residuals)
#   results/         — analysis output CSVs
#   figures/         — output PNGs

set -e
cd "$(dirname "$0")"

STAGE="${1:-all}"

if [[ "$STAGE" == "fetch" || "$STAGE" == "all" ]]; then
    echo "==> Fetching data"
    python -u scripts/00_fetch_data.py
fi

if [[ "$STAGE" == "layer1" || "$STAGE" == "all" ]]; then
    echo "==> Layer 1: season-aggregate static test"
    python -u scripts/snr_test_static.py
fi

if [[ "$STAGE" == "layer2" || "$STAGE" == "all" ]]; then
    echo "==> Layer 2: event studies"
    python -u scripts/hc_midseason_vs_offseason.py
    python -u scripts/oc_change_event_study.py
    python -u scripts/oc_rz_focused_followup.py
    python -u scripts/oc_schematic_isolation.py
    python -u scripts/dc_change_event_study.py
    python -u scripts/dc_rz_focused_followup.py
    python -u scripts/hc_offseason_defense_test.py
fi

if [[ "$STAGE" == "layer3" || "$STAGE" == "all" ]]; then
    echo "==> Layer 3: within-game schematic"
    python -u scripts/snr_test_within_game.py
fi

if [[ "$STAGE" == "layer5" || "$STAGE" == "all" ]]; then
    echo "==> Layer 5: predictive validation + robustness"
    python -u scripts/hc_predictive_validation.py
    python -u scripts/oc_predictive_validation.py
    python -u scripts/predictive_validation_robustness.py
fi

if [[ "$STAGE" == "propensity" || "$STAGE" == "all" ]]; then
    echo "==> Propensity matching (causal rigor check)"
    python -u scripts/hc_propensity_matched.py
fi

if [[ "$STAGE" == "defensive_chronic" || "$STAGE" == "all" ]]; then
    echo "==> §5.3 chronic defensive test (personnel vs coaching identity)"
    python -u scripts/defensive_personnel_vs_coaching_decomp.py
fi

if [[ "$STAGE" == "fingerprint" || "$STAGE" == "all" ]]; then
    echo "==> Coach fingerprint clustering"
    python -u scripts/coach_fingerprint_clustering.py
fi

if [[ "$STAGE" == "svi" || "$STAGE" == "all" ]]; then
    echo "==> SVI-analogue decomposition (team-id vs coaching-history)"
    python -u scripts/snr_test_svi_analogue.py
fi

echo ""
echo "=== DONE ==="
echo "Results: results/"
echo "Figures: figures/"
