#!/bin/bash
#SBATCH --job-name=pvt
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch slurm/run.sbatch <train args>, e.g. --model qwen3.5-2b --layer 12 --mode solo --traits Assertive
#        --source S --out O --run-name R ...   (the arguments of `persvectrain train`; --out and --run-name required)
# Runs `train`, then (if it succeeded) `freegen --out O --run-name R` on vLLM, in two processes so the HF model's
# GPU memory is freed first, then `persona` (vLLM answers + OpenRouter judge; key in OPENROUTER_API_KEY or
# PersVecFast's .env) unless --no-judge. --no-free-gen skips freegen only, --no-judge persona only. Run `mkdir -p slurm-logs` first.
set -euo pipefail
mkdir -p slurm-logs
LIB="${PVT_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib.sh}"
[ -f "$LIB" ] || { echo "submit from the PersVecTrain root or set PVT_LIB" >&2; exit 1; }
source "$LIB"  # sbatch copies this script: find lib.sh via the submit dir
OUT="$(pvt_arg_value --out "$@")"
RUN_NAME="$(pvt_arg_value --run-name "$@")"
pvt_setup_env
echo "$(date)  $(hostname)  python=$PYTHON"
"$PYTHON" -m persvectrain train "$@"
if pvt_has_flag --no-free-gen "$@"; then
  echo "[$(date)] --no-free-gen: skipping freegen"
else
  echo "[$(date)] freegen $RUN_NAME"
  "$PYTHON" -m persvectrain freegen --out "$OUT" --run-name "$RUN_NAME"
fi
if pvt_has_flag --no-judge "$@"; then
  echo "[$(date)] --no-judge: skipping persona"
else
  echo "[$(date)] persona $RUN_NAME"
  "$PYTHON" -m persvectrain persona --out "$OUT" --run-name "$RUN_NAME"
fi
