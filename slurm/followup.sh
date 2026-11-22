#!/bin/bash
# Usage: bash slurm/followup.sh [RUN=JOBID ...]
# The follow-up jobs after the ablation grid, chained so no two jobs write into one run at the same time:
#   per run in RUNS: [RUN's still-running job, given as RUN=JOBID] -> step 1 -> bench
#     step 1: `persona` (adds the length control source_at_trained_norm; saved answers and scores are reused),
#             or, for runs in REEVAL_RUNS, a full re-evaluation (eval + freegen + persona; for an old eval.json)
#     bench:  500 new SuperGPQA questions, incl. the length control
#   and one `vecs --model` job over VEC_RUNS (reads only; logit lens + geometry, incl. the KL-only change)
# RUN=JOBID: the job must finish OK first (afterok); a JOBID that already COMPLETED adds no dependency; one that
# ended otherwise (FAILED, TIMEOUT, ...) skips that run with a message.
# Env: RUNS, REEVAL_RUNS, VEC_RUNS, OUT, SBATCH (default sbatch), SACCT (default sacct).
set -euo pipefail
OUT="${OUT:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
RUNS="${RUNS:-solo_kl0 solo_kl1 solo_kl3 solo_kl5 solo_kl10 solo_kl1_t2048 solo_kl3_t2048 solo_kl5_t2048 solo_kl10_t2048}"
REEVAL_RUNS="${REEVAL_RUNS:-solo_kl0}"
VEC_RUNS="${VEC_RUNS:-solo_kl0 solo_kl1 solo_kl3 solo_kl5 solo_kl10}"
SBATCH="${SBATCH:-sbatch}"
SACCT="${SACCT:-sacct}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PVT_LIB="$HERE/lib.sh"
mkdir -p slurm-logs

declare -A AFTER=()
for kv in "$@"; do
  [[ "$kv" == *=* ]] || { echo "argument $kv is not RUN=JOBID" >&2; exit 1; }
  AFTER["${kv%%=*}"]="${kv#*=}"
done

STEP1_IDS=()
for RUN in $RUNS; do
  DEP=""
  if [ -n "${AFTER[$RUN]:-}" ]; then
    J="${AFTER[$RUN]}"
    STATE="$("$SACCT" -j "$J" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')"
    case "$STATE" in
      PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING) DEP="--dependency=afterok:$J" ;;
      COMPLETED) ;;
      *) echo "$RUN: job $J ended $STATE; skipped (look at slurm-logs/pvt-$J.out)" >&2; continue ;;
    esac
  elif [ ! -f "$OUT/trained/$RUN/train.json" ]; then
    echo "$RUN: no train.json and no RUN=JOBID given; skipped" >&2
    continue
  fi
  if [[ " $REEVAL_RUNS " == *" $RUN "* ]]; then
    S1=$("$SBATCH" --parsable $DEP "$HERE/reeval.sbatch" --out "$OUT" --run-name "$RUN")
    WHAT="re-evaluation"
  else
    S1=$("$SBATCH" --parsable $DEP "$HERE/pvt.sbatch" persona --out "$OUT" --run-name "$RUN")
    WHAT="persona"
  fi
  STEP1_IDS+=("$S1")
  B=$("$SBATCH" --parsable --dependency=afterok:"$S1" "$HERE/bench.sbatch" --out "$OUT" --run-name "$RUN")
  echo "$RUN: ${DEP:+after job ${AFTER[$RUN]}, }$WHAT $S1, then bench $B"
done

# vecs only reads the trained vectors (finished training): no dependency
if [ -n "$VEC_RUNS" ]; then
  V=$("$SBATCH" --parsable "$HERE/vecs.sbatch" --out "$OUT" --runs $VEC_RUNS)
  echo "vecs --model over $VEC_RUNS: $V"
fi
echo "when all have finished:"
echo "python -m persvectrain report --out $OUT --runs $RUNS --per-trait"
