# PersVecTrain

Trains persona steering vectors (one vector per trait at one layer of a frozen Hugging Face model) with
BiPO's persona loss plus a KL term to the unsteered model. It reads the outputs of the sibling repo
PersVecFast (extracted responses and activations) and imports its `persvecfast` package.

## Install

`persvecfast` is not a pip dependency; install it editable from the sibling checkout first:

    pip install -e ../PersVecFast -e ".[test]"

## Commands

- `check` - validate the PersVecFast inputs: kept pairs per trait, dropped traits
- `reference` - unsteered SuperGPQA 200-299 answers of the model (the KL term's reference)
- `train` - train vectors (`--mode solo` or `joint`) and write `train.json`, `steps.jsonl` (with `t`, seconds
  since training start), the vectors and `eval.json`
- `eval` - re-run the evaluation of an existing run
- `report --out O --runs R1 R2 ...` - one pass/fail table over runs and traits, and the run that passes the most
  traits (tie: larger mean trained margin at c = +-1), or "no run passes"

## Cluster run

All paths below are the shared PersVecFast output root. Run from the PersVecTrain checkout with the PersVecFast
environment active.

Install (clone next to the existing code directories):

    git clone <PersVecTrain repo url> PersVecTrain && cd PersVecTrain
    pip install -e ../PersVecFast -e .

Check the inputs:

    python -m persvectrain check --model qwen3.5-2b       --source /scratch/inf0/user/nzukowsk/PersVecFast/out       --traits Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness

Timing job (one epoch, one trait; the `reference` job must finish first):

    REF=$(sbatch --parsable slurm/train.sbatch reference --model qwen3.5-2b       --out /scratch/inf0/user/nzukowsk/PersVecFast/out)
    sbatch --dependency=afterok:$REF slurm/train.sbatch train --model qwen3.5-2b --layer 12 --mode solo       --traits Assertive --epochs 1 --no-free-gen --run-name timing       --source /scratch/inf0/user/nzukowsk/PersVecFast/out --out /scratch/inf0/user/nzukowsk/PersVecFast/out

Seconds per step (over ~10 s per step: stop and report before the sweep):

    python -c "import json,sys; t=[json.loads(l)['t'] for l in open(sys.argv[1])]; print(t[-1]/len(t),'s/step over',len(t),'steps')" /scratch/inf0/user/nzukowsk/PersVecFast/out/trained/timing/steps.jsonl

Sweep (one `reference` job, then four solo runs `solo_kl0`, `solo_kl0.1`, `solo_kl1`, `solo_kl10` after it;
`SOURCE`, `OUT`, `TRAITS` env vars override the defaults):

    bash slurm/sweep.sh qwen3.5-2b 12

Report, once all four have finished:

    python -m persvectrain report --out /scratch/inf0/user/nzukowsk/PersVecFast/out       --runs solo_kl0 solo_kl0.1 solo_kl1 solo_kl10

Joint run at the picked lambda_kl (the value in the picked run's name, e.g. `solo_kl1` gives `KL=1`):

    KL=1
    sbatch slurm/train.sbatch train --model qwen3.5-2b --layer 12 --mode joint --lambda-o 1 --lambda-kl $KL       --traits Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness       --run-name joint_kl$KL       --source /scratch/inf0/user/nzukowsk/PersVecFast/out --out /scratch/inf0/user/nzukowsk/PersVecFast/out
