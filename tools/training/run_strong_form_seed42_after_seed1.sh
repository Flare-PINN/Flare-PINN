#!/usr/bin/env bash
# After seed-1 matched strong-form training finishes, start seed 42 with the same
# MPS auto-restart pattern (train.py exits 42 → resume latest checkpoint in dir).
#
# Prerequisites for seed 42:
#   Place outputs/checkpoints/seed42_pinn_strong_form/checkpoint_step_0042000.pt
#   (your stage-1 / matched 42k for seed 42), OR set STRONG_FORM_SEED42_BOOTSTRAP_CKPT
#   to a .pt file to copy there once before the first launch.
#
# Usage (from repo root):
#   ./tools/training/run_strong_form_seed42_after_seed1.sh
#
# Env:
#   WAIT_FOR_SEED1_STRONG_FORM   default 1 — poll until no train.py using seed1_pinn_strong_form
#   WAIT_PID                      if set, wait for this PID first (overrides the poll above)
#   WANDB_DISABLED                default 1
#   CONDA_ENV_NAME                default flare-pinn
#   SEED42_INITIAL_RESUME         default outputs/checkpoints/seed42_pinn_strong_form/checkpoint_step_0042000.pt
#   STRONG_FORM_SEED42_BOOTSTRAP_CKPT  optional copy source if initial resume file is missing
#   WAIT_POLL_SEC                 default 30

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WAIT_POLL_SEC="${WAIT_POLL_SEC:-30}"
WANDB_DISABLED="${WANDB_DISABLED:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-flare-pinn}"
OUT42="outputs/checkpoints/seed42_pinn_strong_form"
RESUME42="${SEED42_INITIAL_RESUME:-$OUT42/checkpoint_step_0042000.pt}"
BOOTSTRAP="${STRONG_FORM_SEED42_BOOTSTRAP_CKPT:-}"
LOG="outputs/logs/strong_form_seed42_run.log"

mkdir -p "$OUT42" "$(dirname "$LOG")"

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "Waiting for PID ${WAIT_PID} to exit..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${WAIT_POLL_SEC}"
  done
  echo "PID ${WAIT_PID} has exited."
elif [[ "${WAIT_FOR_SEED1_STRONG_FORM:-1}" == "1" ]]; then
  echo "Polling every ${WAIT_POLL_SEC}s until seed-1 strong-form train.py is not running..."
  while pgrep -f "train\.py.*seed1_pinn_strong_form" >/dev/null 2>&1; do
    sleep "${WAIT_POLL_SEC}"
  done
  echo "No matching seed-1 strong-form process found; continuing."
fi

if [[ ! -f "$RESUME42" ]]; then
  if [[ -n "$BOOTSTRAP" && -f "$BOOTSTRAP" ]]; then
    echo "Copying bootstrap checkpoint to ${RESUME42}"
    cp -v "$BOOTSTRAP" "$RESUME42"
  else
    echo "ERROR: initial resume missing: ${RESUME42}"
    echo "Add your seed-42 42k checkpoint there, or set STRONG_FORM_SEED42_BOOTSTRAP_CKPT to a .pt to copy once."
    exit 1
  fi
fi

export WANDB_DISABLED
printf '\n=== strong-form seed42 chain start %s ===\n' "$(date -Iseconds)" | tee -a "$LOG"

RESUME="$RESUME42"
while true; do
  echo ">>> $(date -Iseconds) seed=42 resume=${RESUME}" | tee -a "$LOG"
  conda run -n "${CONDA_ENV_NAME}" --no-capture-output python src/train.py \
    --config src/baselines/strong_form/config_matched.yaml \
    --resume "$RESUME" \
    --seed 42 \
    --checkpoint-dir "$OUT42" \
    2>&1 | tee -a "$LOG"
  ec=${PIPESTATUS[0]}
  if [[ "$ec" -eq 42 ]]; then
    LATEST="$(ls -t "$OUT42"/checkpoint_step_*.pt 2>/dev/null | head -1 || true)"
    if [[ -z "$LATEST" ]]; then
      echo "exit 42 but no checkpoint in ${OUT42}" | tee -a "$LOG"
      exit 1
    fi
    echo "MPS auto-restart: continuing from ${LATEST}" | tee -a "$LOG"
    RESUME="${LATEST}"
    continue
  fi
  exit "$ec"
done
