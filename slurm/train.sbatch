#!/bin/bash
#SBATCH --job-name=pvt
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch slurm/train.sbatch <persvectrain subcommand and args>, e.g. reference --model qwen3.5-2b ...,
# or freegen --out O --run-name R. The vLLM env vars are set, so freegen works here too.
# For train followed by freegen in one job use slurm/run.sbatch. Run `mkdir -p slurm-logs` first.
set -euo pipefail
mkdir -p slurm-logs  # too late for this job's own log: create it before sbatch (README, sweep.sh)
source "${PVT_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib.sh}"  # sbatch copies this script: find lib.sh via the submit dir
pvt_setup_env
echo "$(date)  $(hostname)  python=$PYTHON"
"$PYTHON" -m persvectrain "$@"
