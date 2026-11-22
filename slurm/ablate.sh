#!/bin/bash
# Usage: bash slurm/ablate.sh <model> <layer> [extra sbatch args...]
# The ablation grid lambda_kl x KL window: run `solo_kl{L}` (window 512, today's names) and `solo_kl{L}_t{N}` (other N).
# Per N != 512 whose reference file is missing: one `reference --max-new-tokens N` job. Per cell: a run.sbatch job
# (train + freegen + persona; after its reference job) when trained/<run>/train.json is missing, else a
# reeval.sbatch job (eval + freegen + persona in place, no retraining).
# Env: KL_TOKENS (default "512 2048"), LAMBDAS (default "1 3 5 10"), TRAITS, TRAIN_EXTRA (default "--beta 0.01"),
# SOURCE, OUT (default /scratch/inf0/user/nzukowsk/PersVecFast/out), SBATCH (default sbatch).
set -euo pipefail
MODEL="${1:?Usage: bash slurm/ablate.sh <model> <layer> [extra sbatch args...]}"
LAYER="${2:?Usage: bash slurm/ablate.sh <model> <layer> [extra sbatch args...]}"
shift 2
SOURCE="${SOURCE:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
OUT="${OUT:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
TRAITS="${TRAITS:-Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness}"
TRAIN_EXTRA="${TRAIN_EXTRA---beta 0.01}"
KL_TOKENS="${KL_TOKENS:-512 2048}"
LAMBDAS="${LAMBDAS:-1 3 5 10}"
SBATCH="${SBATCH:-sbatch}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PVT_LIB="$HERE/lib.sh"
mkdir -p slurm-logs

# pvt jobs still queued/running would make "existing run" ambiguous (a train.json of a running job): refuse
SQUEUE="${SQUEUE:-squeue}"
if QUEUED=$("$SQUEUE" -u "${USER:-}" -h -o "%j" 2>/dev/null) && printf '%s
' "$QUEUED" | grep -q '^pvt'; then
  if [ "${FORCE:-}" != "1" ]; then
    echo "pvt jobs are still queued/running; re-run ablate.sh only after they finish (FORCE=1 overrides)" >&2
    exit 1
  fi
fi

run_name() { if [ "$2" = "512" ]; then echo "solo_kl$1"; else echo "solo_kl${1}_t$2"; fi; }
train_finished() {  # status 0 when train.json says "finished": true
  grep -Eq '"finished": *true' "$OUT/trained/$1/train.json"
}

RUNS=""
for N in $KL_TOKENS; do
  DEP=()
  KLFLAG=()
  if [ "$N" != "512" ]; then
    KLFLAG=(--kl-tokens "$N")
    NEED=0
    for L in $LAMBDAS; do [ -f "$OUT/trained/$(run_name "$L" "$N")/train.json" ] || NEED=1; done
    if [ "$NEED" = 1 ] && [ ! -f "$OUT/trained/reference/${MODEL}_pool260_t${N}.jsonl" ]; then
      REF=$("$SBATCH" --parsable "$@" "$HERE/train.sbatch" reference --model "$MODEL" --out "$OUT" --max-new-tokens "$N")
      echo "reference job (window $N): $REF"
      DEP=(--dependency=afterok:"$REF")
    fi
  fi
  for L in $LAMBDAS; do
    RUN="$(run_name "$L" "$N")"
    RUNS="$RUNS $RUN"
    if [ -f "$OUT/trained/$RUN/train.json" ]; then
      if train_finished "$RUN"; then
        ID=$("$SBATCH" --parsable "$@" "$HERE/reeval.sbatch" --out "$OUT" --run-name "$RUN")
        echo "re-evaluation $RUN job: $ID"
      else
        echo "run $RUN not finished (still running or killed): skipped; check squeue / rerun under a new name"
      fi
    else
      # $TRAITS and $TRAIN_EXTRA are split on purpose: one argument per trait / extra word
      # shellcheck disable=SC2086
      ID=$("$SBATCH" --parsable ${DEP[@]+"${DEP[@]}"} "$@" "$HERE/run.sbatch" --mode solo            --model "$MODEL" --layer "$LAYER" --traits $TRAITS --source "$SOURCE" --out "$OUT"            --run-name "$RUN" --lambda-kl "$L" ${KLFLAG[@]+"${KLFLAG[@]}"} $TRAIN_EXTRA)
      echo "train + freegen + persona $RUN job: $ID"
    fi
  done
done
echo "when all have finished:"
echo "python -m persvectrain report --out $OUT --runs$RUNS --per-trait"
