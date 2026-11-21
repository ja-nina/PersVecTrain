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
- `reference` - unsteered SuperGPQA 200-299 answers of the model (the KL term's reference), written as
  `<out>/trained/reference/<model>.jsonl` with its sampling settings and commits in `<model>.meta.json`
- `train` - train vectors (`--mode solo` or `joint`) and write `train.json`, `steps.jsonl`, the vectors and
  `eval.json`
- `eval` - re-run the evaluation of an existing run
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

All paths below are the shared PersVecFast output root. Run from the PersVecTrain checkout with the PersVecFast
environment active. Use PersVecFast's Hugging Face cache, so the model is not downloaded again (the sbatch
script otherwise defaults to `./.cache/huggingface` in the submit directory):

    export HF_HOME=/scratch/inf0/user/nzukowsk/PersVecFast/.cache/huggingface

Install (clone next to the existing code directories):

    git clone <PersVecTrain repo url> PersVecTrain && cd PersVecTrain
    pip install -e ../PersVecFast -e .

Slurm opens `--output=slurm-logs/...` before the job starts, so create the log dir once (`sweep.sh` does it too):

    mkdir -p slurm-logs

Check the inputs:

    python -m persvectrain check --model qwen3.5-2b \
        --source /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --traits Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness

Timing job (one epoch, one trait, with free generation; the `reference` job must finish first). It writes the
reference answers, so the sweep's own `reference` job then finds them and skips:

    REF=$(sbatch --parsable slurm/train.sbatch reference --model qwen3.5-2b \
        --out /scratch/inf0/user/nzukowsk/PersVecFast/out)
    sbatch --dependency=afterok:$REF slurm/train.sbatch train --model qwen3.5-2b --layer 12 --mode solo \
        --traits Assertive --epochs 1 --run-name timing \
        --source /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --out /scratch/inf0/user/nzukowsk/PersVecFast/out

Seconds per training step (over ~10 s per step: stop and report before the sweep):

    python -c "import json,sys; L=[json.loads(l) for l in open(sys.argv[1])]; S=[d for d in L if d['kind']!='start']; print((S[-1]['t']-L[0]['t'])/len(S),'s/step over',len(S),'steps')" \
        /scratch/inf0/user/nzukowsk/PersVecFast/out/trained/timing/steps.jsonl

Eval time: `timing` in `trained/timing/eval.json` (`teacher_forced_s`, `free_gen_s`, seconds), or the
timestamped `eval: ... starts` / `eval: ... done` lines in the job's `slurm-logs/pvt-<job>.out`. The whole run's
wall time is `started` to `ended` in `train.json`. A full solo run is 20 epochs times seven traits of training
plus one eval per trait: check that it fits the 20 h limit of `train.sbatch`.

BiPO saturation, from the start line (`beta * margin` of the start vector; the loss is `-logsigmoid(beta *
margin)`, whose gradient is about `exp(-beta * margin)`, so a large value means almost no persona gradient):

    python -c "import json,sys; d=json.loads(open(sys.argv[1]).readline()); print({s: (r['beta_margin_pos'], r['beta_margin_neg']) for s, r in d['traits'].items()})" \
        /scratch/inf0/user/nzukowsk/PersVecFast/out/trained/timing/steps.jsonl

Rule: if `beta * margin` at the start is above ~5 (at c = +1 or -1), rerun the timing job (new `--run-name`)
with a smaller `--beta` (e.g. `1/margin`) or with `--logprob-norm mean` before the sweep, and run the sweep
with the same setting (pass it to the train jobs, e.g. by adding it in `slurm/sweep.sh`). Per-step `margin` and
`cos_start` in `steps.jsonl` show whether training still moves the vector.

Sweep (one `reference` job, then four solo runs `solo_kl0`, `solo_kl0.1`, `solo_kl1`, `solo_kl10` after it;
`SOURCE`, `OUT`, `TRAITS` env vars override the defaults):

    bash slurm/sweep.sh qwen3.5-2b 12

Report, once all four have finished:

    python -m persvectrain report --out /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --runs solo_kl0 solo_kl0.1 solo_kl1 solo_kl10

Joint run at the picked lambda_kl (the value in the picked run's name, e.g. `solo_kl1` gives `KL=1`):

    KL=1
    sbatch slurm/train.sbatch train --model qwen3.5-2b --layer 12 --mode joint --lambda-o 1 --lambda-kl $KL \
        --traits Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness \
        --run-name joint_kl$KL \
        --source /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --out /scratch/inf0/user/nzukowsk/PersVecFast/out

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
