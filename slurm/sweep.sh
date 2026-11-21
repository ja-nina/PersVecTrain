#!/bin/bash
# Usage: bash slurm/sweep.sh <model> <layer> [extra sbatch args...]
# One reference job, then four solo train jobs (lambda_kl 0, 0.1, 1, 10) that start after it succeeds.
# Env overrides: SOURCE, OUT (default /scratch/inf0/user/nzukowsk/PersVecFast/out), TRAITS (display names).
set -euo pipefail
MODEL="${1:?Usage: bash slurm/sweep.sh <model> <layer> [extra sbatch args...]}"
LAYER="${2:?Usage: bash slurm/sweep.sh <model> <layer> [extra sbatch args...]}"
shift 2
SOURCE="${SOURCE:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
OUT="${OUT:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
TRAITS="${TRAITS:-Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness}"
HERE="$(cd "$(dirname "$0")" && pwd)"

REF=$(sbatch --parsable "$@" "$HERE/train.sbatch" reference --model "$MODEL" --out "$OUT")
echo "reference job: $REF"
RUNS=""
for KL in 0 0.1 1 10; do
  # $TRAITS is split on purpose: one argument per trait
  ID=$(sbatch --parsable --dependency=afterok:"$REF" "$@" "$HERE/train.sbatch" train --mode solo \
       --model "$MODEL" --layer "$LAYER" --traits $TRAITS --source "$SOURCE" --out "$OUT" \
       --run-name "solo_kl$KL" --lambda-kl "$KL")
  echo "train solo_kl$KL job: $ID"
  RUNS="$RUNS solo_kl$KL"
done
echo "when all have finished:"
echo "python -m persvectrain report --out $OUT --runs$RUNS"
