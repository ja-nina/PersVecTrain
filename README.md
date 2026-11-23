# PersVecTrain

Trains persona steering vectors (one vector per trait at one layer of a frozen Hugging Face model) with
BiPO's persona loss plus a KL term to the unsteered model. It reads the outputs of the sibling repo
PersVecFast (extracted responses and activations) and imports its `persvecfast` package.

## Install

`persvecfast` is not a pip dependency; install it editable from the sibling checkout first:

    pip install -e ../PersVecFast -e ".[test]"

## Commands

- `check` - validate the PersVecFast inputs: kept pairs per trait, dropped traits, and each trait's `model` /
  `extraction_model` / `negative_model` from its `{slug}.json`
- `reference` - unsteered answers of the model to the 260-question SuperGPQA pool (records 200-299 + 340-499, `data/supergpqa_pool.jsonl`; 160 KL-train, 100 held-out), the KL term's reference, written as
  `<out>/trained/reference/<model>_pool260.jsonl` with its sampling settings and commits in `<model>_pool260.meta.json`
  (`--max-new-tokens N`, default 512: other N write `<model>_pool260_t{N}.jsonl` / `.meta.json`, for a longer KL window)
- `train` - train vectors (`--mode solo` or `joint`; `--kl-tokens N` reads the `_t{N}` reference and cuts the KL answers to N tokens, default 512; persona pairs stay at `--max-answer-tokens`) and write `train.json`, `steps.jsonl`, the vectors and
  `eval.json`
- `eval` - re-run the evaluation of an existing run
- `freegen --out O --run-name R` - free generation of a finished run on vLLM (unsteered, and per trait trained /
  source_full / trained_random at c = +-1): fills eval.json's `free_gen` block and recomputes the pass rule;
  resumes from its saved answers, `--fresh`, `--rescore-only` (below). `trained_random` is 5 norm-matched random
  directions (files `freegen_vectors/{slug}_trained_random_k0..4.pt`; question i of the held list uses k = i % 5);
  teacher-forced eval averages the same 5 at c = +-1. Answer lines record the vector file name, and saved
  `trained_random` answers without it (old runs) are regenerated on the next `freegen`
- `persona --out O --run-name R [--max-tokens 1024] [--judge-concurrency 32] [--fresh]` - persona answers of a
  finished run on vLLM (unsteered, elicited, and per trait trained / source_full / trained_random at c = +1 neutral
  and c = -1 eliciting prompt) scored by PersVecFast's OpenRouter judge; the key is `OPENROUTER_API_KEY` from the
  environment or `PersVecFast/.env` (checked before anything runs). Fills eval.json's `persona_judge` block and adds
  the judge / coherence checks to the pass rule; resumes (`persona_answers.jsonl.gz`, `persona_judged.jsonl.gz`),
  a judge outage keeps the answers and the rerun judges only what is missing. `train --no-judge` (also `eval`) turns
  it off for a run (status disabled)
- `bench --out O --run-name R [--n N] [--max-tokens] [--fresh]` (`sbatch slurm/bench.sbatch --out O --run-name R`)
  - larger SuperGPQA accuracy check of a finished run on `data/supergpqa_bench500.jsonl`: 500 'hard' questions no
  game / careful-reasoning / KL-pool question overlaps (`scripts/make_bench.py`: the next 500 of the same seed-0
  shuffle as MA_Environments' supergpqa.jsonl). Answered like `freegen` (unsteered; trained / source_full / 5 random
  at c = +-1, paired seeds), resumable (`bench_answers.jsonl.gz`); writes `trained/<run>/bench.json` with the paired
  accuracy differences (trained - unsteered, source - unsteered, trained - source, random - unsteered). Not part of
  the pass rule; `report` prints its table for every listed run that has one. bench and `persona` also answer with
  `source_at_trained_norm` (the source direction scaled to the trained vector's length): the length control, so
  trained - srcN is the effect of the direction alone
- `bash slurm/followup.sh [RUN=JOBID ...]` - after the grid: per run, [after its running job] `persona` (or a full
  re-evaluation for `REEVAL_RUNS`, default solo_kl0) then `bench`, plus one `vecs --model` job; `sbatch
  slurm/pvt.sbatch <subcommand ...>` runs any one command on a GPU node
- `vecs --out O --runs R1 R2 ... [--model] [--n 40]` - analysis, reads only: per trait the change training made
  (trained - start vector; the start is the train-pairs mean difference training begins from), split into
  rescaling along the start and rotation orthogonal to it, against the split noise (source_full - start, a null:
  no training, other pairs); across traits the cosines and 1st-component share of the rotations (shared change?)
  with the split-noise and random baselines, the shared direction u1 and how much of start / trained lies on it;
  across runs the per-trait cosine of the changes. `--model` (GPU, `sbatch slurm/vecs.sbatch --out O --runs ...`)
  adds a logit lens of u1 / the mean change and the geometry of the unsteered layer residuals on the KL reference
  answers (energy in the top 1 / 8 / 64 PCs, variance vs random directions). Writes
  `trained/analysis/vecs_<runs>.json`
- `kldepth --out O --run-name R` - diagnostic, after `freegen`: KL(unsteered || steered) by answer position
  (tokens 0-512, 512-2048, 2048-4096, 4096-8192) on the run's saved unsteered free answers, teacher-forced, per
  trait x trained / source_full / trained_random x c = +-1, plus the paired source_full - trained per bucket.
  Shows whether the 512-token KL term (training's window) also covers the long answers. Writes
  `trained/<run>/kl_depth.json` only; `--max-tokens` (8192), `--n`, `--traits`
- `report --out O --runs R1 R2 ...` - one pass/fail table over runs and traits, and the run that passes the most
  traits (tie: larger mean trained margin at c = +-1), or "no run passes"

Names differing from the spec: the spec's `--steps` is `--epochs` (20 by default). `--min-pairs` (20) drops
traits with fewer kept pairs, in `check` and `train` alike. `--logprob-norm sum|mean` (default `sum`, BiPO's):
`mean` divides each answer's summed log-prob by its answer token count, in training and in the eval margins.

`steps.jsonl` has one line per step (`kind` `single` or `mix`: `loss_p`, `margin` = mean persona margin of the
step's pairs, `loss_kl`, `loss_o`, `norm` and `cos_start` = cosine of v with its start vector, both before the
step, `lr`, `t` = seconds since training start), plus one `kind: "start"` line at the start of each trait (solo)
or of the run (joint): per trait, the mean train persona margin of the start vector at c = +1 (`margin_pos`) and
c = -1 (`margin_neg`) and `beta_margin_pos` / `beta_margin_neg` = beta * margin.

`eval.json` is written by `train` once the teacher-forced part is done (`free_gen.status: "pending"`, the pass
rule with its free-generation checks - length, accuracy and random-length per c - listed in `skipped`) and
rewritten by `freegen` when free generation is done (`status: "done"`, the pass rule recomputed with those
checks); a job killed during free generation leaves it `pending`. `timing.teacher_forced_s` / `timing.free_gen_s`
are the seconds of each part (`free_gen_s` of the last `freegen` job only, when it resumed).

`freegen` appends every engine chunk's answers to `trained/<run>/freegen_answers.jsonl.gz` as soon as the chunk
is done (gzip, one line per answer: `key` = [trait or "-", kind, c, question id], `text`, `n_tokens`,
`finish_reason`, `max_tokens`, `seed`), and scores from that file. Rerunning the same `freegen` (e.g. the same
`sbatch slurm/train.sbatch freegen ...` after a killed or timed-out job) resumes: answers already saved for the
same request (key, seed and `--max-tokens`) are reused, only the rest is generated; a damaged last line (job
killed while writing) is dropped and that answer regenerated. `--fresh` deletes the file first and regenerates
everything. `--rescore-only` loads no model and no engine: it re-parses the saved answers (e.g. after a change to
`parse_answer`) and rewrites eval.json's free-gen parts and pass rule; it needs a saved answer for every request
and uses the `max_tokens` eval.json records unless `--max-tokens` is given.

The answer letter is parsed as MA_Environments' games parse it (`QADebateParser`): the last `ANSWER: X` (any
case, optional markdown around `ANSWER`, `:` or `=` required, optional `*`, `_`, `[`, `(`, `<`, quote or backtick
before the letter), X one of the question's option letters and not followed by another letter; no match = wrong.

### Analysis (`analysis`, notebooks/analysis.ipynb)

`python -m persvectrain analysis --out OUT [--runs R ...] [--z 2] [--no-vectors]` reads every finished run (default:
all under `OUT/trained`) and prints, per metric, run and control, the paired difference trained minus control
averaged over traits (diff > 0 always = trained better), the number of trait cells measurably better / worse
(|z| >= `--z`), every cell where trained is worse than source, and the rotation per run (|trained| / |start|, angle
trained vs start and vs source, the split-noise angle as the null, change along / orthogonal to the start vector,
the share of the change shared across traits). Metrics: bench accuracy and length shift (500 questions), accuracy
on the 100 held questions, judge trait score (c = -1: lower = better) and coherence, teacher-forced margin and KL.
Controls: source, srcN (source direction at the trained length), unsteered, random. Reads only.

`notebooks/analysis.ipynb` shows the same with plots (accuracy / trait / coherence per kind and run, per-trait
heatmaps, the worse-than-source scan, rotation per trait, run-by-run cosines of the trained vectors and of the
changes, the logit lens from saved `vecs --model` files, rotation angle vs gain over srcN). It needs pandas,
matplotlib and nbconvert; run it from the repo root with
`PVT_OUT=OUT jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb` (`PVT_RUNS="a b"`
limits the runs). The notebook is generated by `scripts/make_analysis_notebook.py`; edit the cells there.

## Cluster run

All paths are the shared PersVecFast output root. Run everything from the PersVecTrain checkout. `train` runs on
Hugging Face (teacher-forced evaluation); free generation is a separate `freegen` step on vLLM (PersVecFast's
steer engine), run in the same job by `slurm/run.sbatch` (separate processes, so the HF model's GPU memory is freed
first), followed by `persona` unless `--no-judge`. A run is `pending` until its `freegen` and `persona` have finished: `report` shows it as PENDING and never picks it.

Update:

    git pull
    pip install -e ../PersVecFast -e .

Environment (PersVecFast's Hugging Face cache, so the model is not downloaded again; Slurm opens the log file
before the job starts, so the log dir must exist):

    micromamba activate pvf
    export HF_HOME=/scratch/inf0/user/nzukowsk/PersVecFast/.cache/huggingface
    mkdir -p slurm-logs

Reference answers for the 260-question pool (the KL term's reference; the sweep's own `reference` job then finds
them and skips):

    REF=$(sbatch --parsable slurm/train.sbatch reference --model qwen3.5-2b \
        --out /scratch/inf0/user/nzukowsk/PersVecFast/out)

Timing job: one trait, one epoch, free generation on, after the reference job:

    sbatch --dependency=afterok:$REF slurm/run.sbatch --model qwen3.5-2b --layer 12 --mode solo \
        --traits Assertive --epochs 1 --beta 0.01 --run-name timing_vllm \
        --source /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --out /scratch/inf0/user/nzukowsk/PersVecFast/out

Reading it (job log `slurm-logs/pvt-<job>.out`, and `trained/timing_vllm/eval.json`):

- the free-generation time: `timing.free_gen_s` in `eval.json` (seconds), or the log line
  `freegen timing_vllm: done in <s> s`; `timing.teacher_forced_s` is the HF evaluation, and `started` / `ended`
  in `train.json` the whole training
- the free-gen table (tokens, cut share and accuracy per trait x kind x c, then the pass line per trait) is
  printed at the end of the log, after the `freegen timing_vllm: done` line
- seconds per training step: `t` in `trained/timing_vllm/steps.jsonl` (stop and report if over ~10 s per step)
- BiPO saturation: `beta_margin_pos` / `beta_margin_neg` in the first line of `steps.jsonl` (`beta * margin` of
  the start vector; above ~5 almost no persona gradient): rerun with a smaller `--beta` under a new `--run-name`
  before the sweep

Sweep: one `reference` job, then four solo runs `solo_kl0`, `solo_kl0.1`, `solo_kl1`, `solo_kl10` (each train +
freegen in one job) after it. `SOURCE`, `OUT`, `TRAITS` env vars override the defaults; `TRAIN_EXTRA` is appended
to every train's arguments:

    TRAIN_EXTRA="--beta 0.01" bash slurm/sweep.sh qwen3.5-2b 12

It prints the job ids and the report command. Report, once all four have finished:

    python -m persvectrain report --out /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --runs solo_kl0 solo_kl0.1 solo_kl1 solo_kl10

Joint run at the picked lambda_kl (the value in the picked run's name, e.g. `solo_kl1` gives `KL=1`):

    KL=1
    sbatch slurm/run.sbatch --model qwen3.5-2b --layer 12 --mode joint --beta 0.01 --lambda-o 1 --lambda-kl $KL \
        --traits Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness \
        --run-name joint_kl$KL \
        --source /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --out /scratch/inf0/user/nzukowsk/PersVecFast/out

`freegen` alone (an existing run whose status is `pending`, e.g. after a failed or timed-out free generation)
runs through the generic runner, which sets the same vLLM environment; it resumes from the run's
`freegen_answers.jsonl.gz` (add `--fresh` to regenerate everything):

    sbatch slurm/train.sbatch freegen --out /scratch/inf0/user/nzukowsk/PersVecFast/out --run-name solo_kl1

Re-parse the saved answers without generating (no model, no vLLM, no GPU: run it directly in the env):

    python -m persvectrain freegen --out /scratch/inf0/user/nzukowsk/PersVecFast/out --run-name solo_kl1 \
        --rescore-only

KL by answer position (`kldepth`, HF on one GPU, after `freegen`):

    sbatch slurm/train.sbatch kldepth --out /scratch/inf0/user/nzukowsk/PersVecFast/out --run-name solo_kl1

`report` shows a pending run's traits as PENDING, or `PENDING (FAIL: <checks>)` when a teacher-forced check
already fails (free generation cannot make it pass; random@c counts only when its KL part failed); a pending run is never picked.
An eval.json written before the current code (no `persona_judge` key, or free generation without `random_dirs`) is
OLD: shown as `OLD` in every cell, listed under the table with the command
`sbatch slurm/reeval.sbatch --out O --run-name R`, and never picked.

`--no-free-gen` on `run.sbatch` (or `train`) skips free generation: the run's free_gen status is `disabled`, its
length and accuracy checks are skipped (listed in `skipped`) and `report` shows it normally. Re-running `eval`
on such a run without `--no-free-gen` makes it `pending` again. A run trained before this change (status
`pending` from the old Hugging Face path, or with Hugging Face free-gen results) must be rerun under a new
`--run-name`.

## Ablation

The grid lambda_kl (1, 3, 5, 10) x KL window (512, 2048 answer tokens): runs `solo_kl{L}` (window 512, the existing
runs) and `solo_kl{L}_t2048`. One script submits it: a `reference --max-new-tokens 2048` job when its file is
missing, a train + freegen + persona job (`run.sbatch`, after the reference job for the 2048 runs) for every run
without a `train.json`, and a `reeval.sbatch` job (`eval`, `freegen`, `persona` in place, no retraining) for every
existing run. `KL_TOKENS`, `LAMBDAS`, `TRAITS`, `TRAIN_EXTRA` (default `--beta 0.01`), `SOURCE`, `OUT` override the
defaults. Re-run `ablate.sh` only after earlier jobs have finished: it refuses while `pvt` jobs are in the
queue (`FORCE=1` overrides), skips a run whose `train.json` says `finished: false` (still training or killed; check
`squeue`, rerun under a new name) and re-evaluates only finished runs.

    bash slurm/ablate.sh qwen3.5-2b 12 -p gpu-l40 --gres=gpu:1 -c 10 --mem=64G

When all jobs have finished (the script prints this command with every run), the table plus the best run per trait,
and a profile `{trait: {run, vector}}` of the picked runs' vector files (null for a trait no run passes):

    python -m persvectrain report --out /scratch/inf0/user/nzukowsk/PersVecFast/out         --runs solo_kl1 solo_kl3 solo_kl5 solo_kl10 solo_kl1_t2048 solo_kl3_t2048 solo_kl5_t2048 solo_kl10_t2048         --per-trait --write-profile profile.json

`persona` alone (vLLM answers and the judge, for a run whose persona step failed or was skipped; it resumes and
only judges what is missing):

    sbatch slurm/train.sbatch persona --out /scratch/inf0/user/nzukowsk/PersVecFast/out --run-name solo_kl1

The judge needs `OPENROUTER_API_KEY` in the environment or in PersVecFast's `.env`; `persona` checks it before
generating anything.

## Using the output in MA_Environments

For the games (stage 3), steer from the run's vector dir:

- `vectors_dir: /scratch/inf0/user/nzukowsk/PersVecFast/out/trained/<run_name>/persona/<model>/bf16`
- `traits_dir: /scratch/inf0/user/nzukowsk/PersVecFast/data/persona_vectors` (the trait instructions are not
  copied into the run dir)
- a fixed `layer: 12`: only that layer of each `{trait}.pt` is trained, the other layers are the source vectors
- a `logging_dir` of its own, not the one of the source-vector runs: cached 2-trait mixes in
  `<logging_dir>/_vectors/<model>` are named without a vector fingerprint, so a shared dir would reuse the
  source vectors' mixes
- the monitor reads its trait axis from the same `vectors_dir`, i.e. the trained vectors at L12; to compare trait
  scores with the source-vector runs, point the monitor at the source vectors dir (or score at layers other
  than 12, which hold the source vectors)
- traits dropped by `train` (too few kept pairs, see `dropped_traits` in `train.json`) are absent from the dir
