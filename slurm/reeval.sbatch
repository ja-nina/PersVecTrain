#!/bin/bash
#SBATCH --job-name=pvt-reeval
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch slurm/reeval.sbatch --out O --run-name R
# Re-evaluates an existing run in place, without retraining: `eval`, then `freegen` (vLLM), then `persona` (vLLM
# answers + OpenRouter judge; key in OPENROUTER_API_KEY or PersVecFast's .env). Run `mkdir -p slurm-logs` first.
set -euo pipefail
mkdir -p slurm-logs
LIB="${PVT_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib.sh}"
[ -f "$LIB" ] || { echo "submit from the PersVecTrain root or set PVT_LIB" >&2; exit 1; }
source "$LIB"
OUT="$(pvt_arg_value --out "$@")"
RUN_NAME="$(pvt_arg_value --run-name "$@")"
pvt_setup_env
echo "$(date)  $(hostname)  python=$PYTHON"
echo "[$(date)] eval $RUN_NAME"
"$PYTHON" -m persvectrain eval --out "$OUT" --run-name "$RUN_NAME"
echo "[$(date)] freegen $RUN_NAME"
"$PYTHON" -m persvectrain freegen --out "$OUT" --run-name "$RUN_NAME"
echo "[$(date)] persona $RUN_NAME"
"$PYTHON" -m persvectrain persona --out "$OUT" --run-name "$RUN_NAME"
