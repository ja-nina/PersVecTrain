#!/usr/bin/env bash
# Collect everything the write-up needs into one archive: per-run json files (train, eval, bench, offtarget,
# judge lines, steps), the full analysis report, the pass report, answer patterns + example pairs of every run with
# a finished bench, the vecs analysis files, the notebook figures, squeue and git state. Reads only (writes the
# bundle dir and its .tgz). Runs one heavy step at a time (the submit node kills parallel memory use).
#   nohup bash scripts/make_bundle.sh bundle_1124 > bundle_1124.log 2>&1 &
B=${1:-bundle}
OUT=${OUT:-/scratch/inf0/user/nzukowsk/PersVecFast/out}
rm -rf "$B" "$B.tgz"
mkdir -p "$B/runs" "$B/patterns"
RUNS=$(ls "$OUT/trained" | grep -E '^solo_')
echo "$RUNS" > "$B/runs.txt"
echo "[$(date +%T)] runs: $(echo $RUNS)"

for r in $RUNS; do
  d="$OUT/trained/$r"
  mkdir -p "$B/runs/$r"
  for f in train.json eval.json bench.json offtarget.json steps.jsonl kl_depth.json \
           persona_judged.jsonl.gz offtarget_judged.jsonl.gz; do
    [ -f "$d/$f" ] && cp "$d/$f" "$B/runs/$r/"
  done
  ls -la "$d" > "$B/runs/$r/ls.txt"
done
[ -d "$OUT/trained/analysis" ] && cp -r "$OUT/trained/analysis" "$B/vecs_analysis"
echo "[$(date +%T)] copied run files"

python -m persvectrain analysis --out "$OUT" > "$B/analysis.txt" 2> "$B/analysis.err"
echo "[$(date +%T)] analysis: $(wc -l < "$B/analysis.txt") lines"
python -m persvectrain report --out "$OUT" --runs $RUNS > "$B/report.txt" 2>&1
echo "[$(date +%T)] report done"

for r in $RUNS; do
  if [ -f "$OUT/trained/$r/bench.json" ]; then
    python scripts/answer_patterns.py --answers "$OUT/trained/$r/bench_answers.jsonl.gz" --name "$r" \
      --out-dir "$B/patterns" > /dev/null 2>> "$B/patterns.err" || echo "patterns $r: FAILED" >> "$B/patterns.err"
    echo "[$(date +%T)] patterns $r"
  else
    echo "patterns $r: no bench.json yet, skipped" >> "$B/patterns.err"
  fi
done

PVT_OUT="$OUT" PVT_RUNS="$(echo $RUNS)" python scripts/run_notebook.py --dir "$B/figures" > "$B/notebook.log" 2>&1
echo "[$(date +%T)] figures: $(ls "$B/figures" 2>/dev/null | wc -l) files"

squeue -u "$USER" -o "%.10i %.12j %.10P %.2t %.10M %.20R" > "$B/squeue.txt" 2>&1
git log --oneline -5 > "$B/git.txt"
tar czf "$B.tgz" "$B"
echo "[$(date +%T)] done:"
ls -la "$B.tgz"
du -sh "$B"/* | sort -h | tail -8
cat "$B/patterns.err" "$B/analysis.err" 2>/dev/null | grep -v "^analysis: skipped timing" | head -20
