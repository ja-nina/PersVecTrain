#!/bin/bash
#SBATCH --job-name=pvt-vecs
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch slurm/vecs.sbatch --out O --runs R1 R2 ... [--n 40]
# `vecs --model`: the vector analysis plus logit lens and layer residual geometry (needs the model on GPU).
set -euo pipefail
mkdir -p slurm-logs
LIB="${PVT_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib.sh}"
[ -f "$LIB" ] || { echo "submit from the PersVecTrain root or set PVT_LIB" >&2; exit 1; }
source "$LIB"
pvt_setup_env
echo "$(date)  $(hostname)  python=$PYTHON"
"$PYTHON" -m persvectrain vecs --model "$@"
