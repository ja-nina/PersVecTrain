#!/bin/bash
#SBATCH --job-name=pvt-cmd
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch slurm/pvt.sbatch <persvectrain subcommand and its arguments>
# e.g. sbatch slurm/pvt.sbatch persona --out O --run-name R   (vLLM env as run.sbatch). Run `mkdir -p slurm-logs` first.
set -euo pipefail
mkdir -p slurm-logs
LIB="${PVT_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib.sh}"
[ -f "$LIB" ] || { echo "submit from the PersVecTrain root or set PVT_LIB" >&2; exit 1; }
source "$LIB"
pvt_setup_env
echo "$(date)  $(hostname)  python=$PYTHON  args: $*"
"$PYTHON" -m persvectrain "$@"
