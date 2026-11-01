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
  snr_test_within_game defensive_personnel_vs_coaching_decomp coach_fingerprint_clustering snr_test_svi_analogue"
WINDOW_2025="hc_predictive_validation oc_predictive_validation predictive_validation_robustness hc_propensity_matched"

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
done

echo ""
"$PY" scripts/build/check_results.py "$MODE"
