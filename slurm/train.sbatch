#!/bin/bash
#SBATCH --job-name=pvt
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/pvt-%j.out
# Usage: sbatch slurm/train.sbatch <persvectrain subcommand and args>, e.g. train --model qwen3.5-2b --layer 12 ...
set -euo pipefail
mkdir -p slurm-logs
PYTHON="${PYTHON:-$(command -v python || true)}"  # active env at sbatch time (sbatch exports PATH)
[ -x "$PYTHON" ] || { echo "no python found: activate the env (with persvecfast and persvectrain installed) before sbatch" >&2; exit 1; }
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
echo "$(date)  $(hostname)  python=$PYTHON"
"$PYTHON" -m persvectrain "$@"
