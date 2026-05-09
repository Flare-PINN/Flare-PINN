#!/usr/bin/env bash
# Re-run test evaluation with validation-frozen D2C thresholds and export POD/FPR/FAR/CSI
# (via validate_checkpoint.py + compute_confusion_matrix.py).
#
# Usage (from repo root):
#   bash tools/validation/rerun_test_d2c_metrics.sh              # all three models
#   bash tools/validation/rerun_test_d2c_metrics.sh weak         # weak-form only (common)
#   METRICS_MODEL=weak bash tools/validation/rerun_test_d2c_metrics.sh
#
# METRICS_MODEL (or first arg): all | benchmark | weak | strong
#
# Optional:
#   TEST_PARQUET=data/windows_test_15.parquet OUT_DIR=final_results/metrics bash ...

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

# Use a function instead of a word-array for `conda run … python` — arrays + line
# continuations can split wrong in some shells and make the next line execute as a bare path
# (e.g. ".pt: Permission denied").
py() {
  if command -v conda >/dev/null 2>&1; then
    conda run -n flare-pinn --no-capture-output python "$@"
  else
    command python "$@"
  fi
}

TEST_PARQUET="${TEST_PARQUET:-data/windows_test_15.parquet}"
OUT_DIR="${OUT_DIR:-final_results/metrics}"
mkdir -p "${OUT_DIR}"

# First CLI arg overrides METRICS_MODEL; default = full sweep (paper-style table).
MODE="${1:-${METRICS_MODEL:-all}}"
case "${MODE}" in
  all|full) MODE=all ;;
  benchmark|cnn|baseline|bench) MODE=benchmark ;;
  weak|weak-form|pinn) MODE=weak ;;
  strong|strong-form) MODE=strong ;;
  *)
    echo "Unknown mode: ${MODE}. Use: all | benchmark | weak | strong" >&2
    exit 1
    ;;
esac

run_benchmark() {
  echo "== CNN-GRU benchmark (40k) =="
  py tools/validation/validate_checkpoint.py \
    --config src/configs/benchmark_classifier.yaml \
    --checkpoint outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt \
    --data "${TEST_PARQUET}"
  py tools/validation/compute_confusion_matrix.py \
    --input outputs/checkpoints/benchmark_classifier/validation_results/checkpoint_step_0040000_test.npz \
    --model-name "CNN-GRU benchmark" \
    --output-dir "${OUT_DIR}" \
    --csv-name benchmark_40k_confusion_metrics.csv
}

run_weak() {
  echo "== Weak-form PINN (44k, locked selection) =="
  py tools/validation/validate_checkpoint.py \
    --config src/configs/flare_pinn_final.yaml \
    --checkpoint outputs/checkpoints/weak_form/final/checkpoint_step_0044000.pt \
    --data "${TEST_PARQUET}"
  py tools/validation/compute_confusion_matrix.py \
    --input outputs/checkpoints/weak_form/final/validation_results/checkpoint_step_0044000_test.npz \
    --model-name "Weak-form PINN" \
    --output-dir "${OUT_DIR}" \
    --csv-name weak_pinn_44k_confusion_metrics.csv
}

run_strong() {
  echo "== Strong-form PINN (44k) =="
  py tools/validation/validate_checkpoint.py \
    --config src/baselines/strong_form/config_matched.yaml \
    --checkpoint "outputs/checkpoints/Strong Form Pinn Final 44k/checkpoint_step_0044000.pt" \
    --data "${TEST_PARQUET}"
  py tools/validation/compute_confusion_matrix.py \
    --input "outputs/checkpoints/Strong Form Pinn Final 44k/validation_results/checkpoint_step_0044000_test.npz" \
    --model-name "Strong-form PINN" \
    --output-dir "${OUT_DIR}" \
    --csv-name strong_pinn_44k_confusion_metrics.csv
}

case "${MODE}" in
  all)
    run_benchmark
    run_weak
    run_strong
    ;;
  benchmark) run_benchmark ;;
  weak) run_weak ;;
  strong) run_strong ;;
esac

echo "Done. CSVs under ${OUT_DIR}/"
