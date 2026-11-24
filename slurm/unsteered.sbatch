#!/bin/bash
#SBATCH --job-name=pvt-unst
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch [-p gpu-a40 --exclude=...] slurm/unsteered.sbatch --out O --model-from R --tag T --seeds 1 2 3
# Extra unsteered draws of the 500-question bench (scripts/unsteered_draws.py); resumable. One --tag per GPU type.
set -euo pipefail
mkdir -p slurm-logs
LIB="${PVT_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib.sh}"
[ -f "$LIB" ] || { echo "submit from the PersVecTrain root or set PVT_LIB" >&2; exit 1; }
source "$LIB"
pvt_setup_env
echo "$(date)  $(hostname)  python=$PYTHON  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
"$PYTHON" scripts/unsteered_draws.py "$@"
