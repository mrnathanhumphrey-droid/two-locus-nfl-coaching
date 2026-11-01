#!/usr/bin/env bash
# Reproduce the two-locus NFL coaching results from the frozen inputs in data/frozen.
#
#   bash reproduce.sh              # the submitted paper's numbers (same as: bash reproduce.sh submitted)
#   bash reproduce.sh corrected    # the same analysis with every team under one team-code standard
#   bash reproduce.sh verify       # rebuild the panels from data/frozen with the original builders, prove equal
#
# Outputs land in results/<mode>/ and are checked against results/expected/<mode>/ at the end.
# No download and no scraping: everything runs from data/frozen (see data/frozen/SOURCES.md).
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-submitted}"
PY="${PYTHON:-python}"

case "$MODE" in
  submitted|corrected) ;;
  verify) exec "$PY" -u scripts/build/panels.py submitted ;;
  *) echo "usage: bash reproduce.sh [submitted|corrected|verify]"; exit 2 ;;
esac

(cd data/frozen && "$PY" -c "import hashlib, sys
bad = [l.split()[1] for l in open('MANIFEST.sha256') if l.strip()
       and hashlib.sha256(open(l.split()[1], 'rb').read()).hexdigest() != l.split()[0]]
sys.exit('frozen inputs differ from MANIFEST.sha256: ' + ', '.join(bad) if bad else 0)")
echo "==> frozen inputs match MANIFEST.sha256"

if [[ "$MODE" == "corrected" ]]; then
  echo "==> building corrected panels"
  "$PY" -u scripts/build/panels.py corrected > /dev/null
fi

# the two data windows the study ran on (see scripts/build/stage.py)
WINDOW_2024="snr_test_static hc_midseason_vs_offseason oc_change_event_study oc_rz_focused_followup
  oc_schematic_isolation dc_change_event_study dc_rz_focused_followup hc_offseason_defense_test
  snr_test_within_game coach_fingerprint_clustering snr_test_svi_analogue"
# defensive_personnel_vs_coaching_decomp gives identical results in either window; it runs here because the
# Sloan revision analyses read its defensive outcome panel
WINDOW_2025="hc_predictive_validation oc_predictive_validation predictive_validation_robustness hc_propensity_matched
  defensive_personnel_vs_coaching_decomp"
# the Sloan revision analyses (scripts/revision), in dependency order, on the through-2025 data
REVISION="oc_concept_family_portability fit_rc_event_study_pymc rc_propensity_refit rc_placebo_and_propensity
  dc_within_hc_event_study build_extended_defensive_panel dc_within_hc_extended dc_within_hc_era_strat make_sloan_figures"
# the NBA transfer check (scripts/nba): same framework, nba.com data in data/frozen/nba
NBA="build_schematic_residuals layer1_partition layer2_midseason_vs_offseason layer2_offseason_hc_event_study
  layer2_defensive_analog hc_predictive_validation opp_3pt_predictive_validation propensity_with_roster
  nextgen_event_study nextgen_propensity coach_fingerprint_clustering"

OUT="results/$MODE"; rm -rf "$OUT"; mkdir -p "$OUT"
for W in 2024 2025; do
  "$PY" -u scripts/build/stage.py "$MODE" "$W"
  LIST="WINDOW_$W"
  for s in ${!LIST}; do
    echo "==> [$MODE, through $W] $s"
    "$PY" -u "scripts/$s.py" > "$OUT/$s.log" 2>&1 || { echo "   FAILED, see $OUT/$s.log"; exit 1; }
  done
  # collect this window's outputs before the next stage clears the layout
  cp data/*.csv "$OUT"/ 2>/dev/null || true
  cp data/nfl/*.csv "$OUT"/ 2>/dev/null || true
  if [[ "$W" == "2025" ]]; then
    cp data/*.csv data/nfl/        # the revision scripts read these from data/nfl, as the research layout kept them
    mkdir -p data/nfl/sloan_revision_outputs "$OUT/revision"
    for s in $REVISION; do
      echo "==> [$MODE, revision] $s"
      "$PY" -u "scripts/revision/$s.py" > "$OUT/revision/$s.log" 2>&1 || { echo "   FAILED, see $OUT/revision/$s.log"; exit 1; }
    done
    cp data/nfl/sloan_revision_outputs/*.{csv,json,parquet} "$OUT/revision/"
    cp data/nfl/sloan_revision_outputs/*.png figures/    # Figure 1 of the extended abstract
  fi
done

rm -rf data/nba; mkdir -p data/nba/data/results data/nba/data/parquet "$OUT/nba"
for s in $NBA; do
  echo "==> [$MODE, nba] $s"
  "$PY" -u "scripts/nba/$s.py" > "$OUT/nba/$s.log" 2>&1 || { echo "   FAILED, see $OUT/nba/$s.log"; exit 1; }
done
cp data/nba/data/results/*.csv data/nba/data/parquet/*.parquet "$OUT/nba/"

echo ""
if [[ "$MODE" == "submitted" ]]; then
  "$PY" scripts/build/check_abstract.py || true    # every abstract figure vs its reproduced value (a report)
  echo ""
fi
"$PY" scripts/build/check_results.py "$MODE"
