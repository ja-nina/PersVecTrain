#!/bin/bash
#SBATCH --job-name=pvt-bench
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch slurm/bench.sbatch --out O --run-name R [--n N]
# The larger SuperGPQA accuracy check (`bench`, 500 new questions) of a finished run; resumes from
# bench_answers.jsonl.gz if a job is cut. Run `mkdir -p slurm-logs` first.
set -euo pipefail
mkdir -p slurm-logs
LIB="${PVT_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/lib.sh}"
[ -f "$LIB" ] || { echo "submit from the PersVecTrain root or set PVT_LIB" >&2; exit 1; }
source "$LIB"
pvt_setup_env
echo "$(date)  $(hostname)  python=$PYTHON"
"$PYTHON" -m persvectrain bench "$@"
