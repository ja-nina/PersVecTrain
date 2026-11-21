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
- `train` - train vectors (`--mode solo` or `joint`) and write `train.json`, `steps.jsonl`, the vectors and
  `eval.json`
- `eval` - re-run the evaluation of an existing run
- `freegen --out O --run-name R` - free generation of a finished run on vLLM (unsteered, and per trait trained /
  source_full / trained_random at c = +-1): fills eval.json's `free_gen` block and recomputes the pass rule
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

`eval.json` is written once the teacher-forced part is done (`free_gen.status: "pending"`, the pass rule with its
length checks listed in `skipped`) and rewritten when free generation is done (`status: "done"`); a job killed
during free generation keeps the first one. `timing.teacher_forced_s` / `timing.free_gen_s` are the seconds of
each part.

## Cluster run

All paths are the shared PersVecFast output root. Run everything from the PersVecTrain checkout. `train` runs on
Hugging Face (teacher-forced evaluation); free generation is a separate `freegen` step on vLLM (PersVecFast's
steer engine), run in the same job by `slurm/run.sbatch` (two processes, so the HF model's GPU memory is freed
first). A run is `pending` until its `freegen` has finished: `report` shows it as PENDING and never picks it.

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

`freegen` alone (an existing run whose status is `pending`, e.g. after a failed free generation) runs through the generic runner, which sets the same vLLM environment:

    sbatch slurm/train.sbatch freegen --out /scratch/inf0/user/nzukowsk/PersVecFast/out --run-name solo_kl1

`--no-free-gen` on `run.sbatch` (or `train`) skips free generation: the run's free_gen status is `disabled`, its
length and accuracy checks are skipped (listed in `skipped`) and `report` shows it normally. Re-running `eval`
on such a run without `--no-free-gen` makes it `pending` again. A run trained before this change (status
`pending` from the old Hugging Face path, or with Hugging Face free-gen results) must be rerun under a new
`--run-name`.

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
