# Ablation grid, 5 random directions, longer KL answers, persona judge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The user can run an ablation grid (KL window 512 / 2048 tokens x KL weight 1, 3, 5, 10), judge persona
expression of every run with the OpenRouter judge, compare against 5 random directions instead of one, and pick the
best run per trait.

**Architecture:** Same repo, same commands. `evaluate` / `freegen` get a 5-direction random set; `reference` and
`train` get a KL window option; a new `persona` command (vLLM generation + PersVecFast's OpenRouter judge) adds a
`persona_judge` block to eval.json and two checks to the pass rule; `report --per-trait` picks a run per trait;
`slurm/ablate.sh` submits the grid and re-evaluates runs that already exist.

**Tech Stack:** as before; PersVecFast `persvecfast.judge` (httpx, OpenRouter, `openai/gpt-oss-120b`).

**Spec:** `docs/superpowers/specs/2026-11-20-trained-steering-vectors-design.md`. Each task updates the spec
sections it changes (Evaluation, Pass rule, Free-generation check, new "Persona judge" and "Ablation" sections).

## Why (sweep, 2026-11-22)

solo_kl{0,0.1,1,3,10} at beta 0.01: weight 1 fixes KL only in the first 512 tokens (kldepth: the gain fades by
2-4k tokens and reverses past 4k), weight 10 fixes it at every depth but loses the persona, weight 3 passes
sarcastic. The single random twin (seed 0 + layer, the same direction for every trait) makes answers long at
c = -1 for every trait. The user chose: longer KL answers AND per-trait weight selection, as separate ablation
runs; the random check against the mean of 5 random directions; persona expression checked with the OpenRouter
judge.

## Global Constraints

- Random set: 5 directions, `random_seeds(layer) = [RANDOM_CONTROL_SEED + layer + 1000 * k for k in range(5)]`
  (k = 0 is the current twin), each `random_control(v, seed)` (norm-matched to v). Kind names stay
  `trained_random` / `source_random`; eval.json records `"random_dirs": 5`.
- KL window: `reference --max-new-tokens N` (default 512). N = 512 keeps today's file name
  `<model>_pool260.jsonl`; other N write `<model>_pool260_t{N}.jsonl` (+ `.meta.json`). `TrainConfig.kl_tokens`
  (default 512, `--kl-tokens`) picks the file and truncates the KL answers (train and eval); persona pairs stay at
  `max_answer_tokens`. Old train.json without `kl_tokens` reads as 512.
- Persona judge: PersVecFast `persvecfast.judge` (`Judges`, `session`, `score_expression`, `score_coherence`,
  `load_config`), expression model `openai/gpt-oss-120b`, temperature 0. `OPENROUTER_API_KEY` from the env or
  PersVecFast's `.env` (`load_dotenv(PERSVECFAST_REPO / ".env")`), checked before any generation.
- Persona questions: the union of every trait's held pair questions of the run (train.json splits), ordered by
  trait (cfg.traits order) then pair index; the same list for every trait.
- Persona conditions per trait: `unsteered` (neutral system prompt) and `elicited` (PersVecFast
  `positive_instruction(trait, False)` as system prompt, no steering); `trained`, `source_full`, `trained_random`
  at c = +1 under the neutral prompt and at c = -1 under the eliciting prompt (as PersVecFast's suppression).
  Thinking off. Max tokens 1024 (`--max-tokens`). Seeds `request_seed(cfg.seed, "persona", c, qi)` for steered
  (paired across kinds and traits), `request_seed(cfg.seed, "persona", "neutral" | "elicited", qi)` unsteered.
- Random rotation in free answers (freegen and persona): the item at position i of the question list uses random
  direction k = i % 5; vector files `run_root/freegen_vectors/{slug}_trained_random_k{k}.pt`.
- Pass rule additions, at c = +1 and -1, paired, 1-SE ties: `judge@+1`: mean(score_trained - score_source) >= -SE;
  `judge@-1`: mean(score_source - score_trained) >= -SE (trained suppresses at least as much);
  `coherence@c`: mean(coherence_trained - coherence_source) >= -SE. Skipped (listed in `skipped`) without judge
  results. The random check is unchanged in form, now on the 5-direction means.
- Never write into the source dir; never change an existing run's vectors or train.json from eval / freegen /
  persona.

## Review Focus

1. Resume safety: a saved free answer made with the old single random direction must never be reused for a
   rotated direction (answer lines carry the vector file name).
2. An existing run (no `persona_judge` key, old single-random eval.json) must still load in `report` and be
   re-evaluated in place by `eval` -> `freegen` -> `persona` without retraining.
3. A judge outage (no credit, bad key) stops `persona` with a clear message and keeps the generated answers; a
   rerun only judges what is missing.
4. Pairing: same question order and seed for every kind of one (trait, c) in persona and freegen.
5. A missing `_t2048` reference makes `train --kl-tokens 2048` fail before loading the model, naming the command
   that makes it.

---

### Task 1: 5 random directions (evaluate + freegen)

**Files:** `src/persvectrain/evaluate.py`, `src/persvectrain/freegen.py`, tests, spec (Evaluation, Pass rule).

- Produces: `evaluate.random_seeds(layer) -> list[int]`; `evaluate.random_set(v, layer) -> list[Tensor]`.
  Teacher-forced: per item, `trained_random` / `source_random` margin and KL are the mean over the 5 directions at
  c = +-1 (the pass-rule c values); at c = +-0.5 only k = 0 (cost). Store per direction means under
  `kinds[...][c]["per_direction"]` (list of 5 kl means) at c = +-1. eval.json `"random_dirs": 5`.
- freegen: `write_freegen_vectors` writes `{slug}_trained_random_k{k}.pt` for k in 0..4 (no plain
  `{slug}_trained_random.pt` any more); the trained_random request for held position i uses k = i % 5; answer
  lines gain `"vector": <file name>`; `saved_results` reuses a trained_random line only when its `vector` equals
  the expected file name (lines without it never), other kinds as before. Per-trait free-gen block for
  trained_random gains `"direction": [k per item]`.
- `eval` on an existing run keeps its free_gen and (Task 3) persona_judge blocks as today.
- Tests: seeds list; random_set norms equal |v| and differ from each other; teacher-forced random value = mean of
  the 5 per-direction values (tiny model); freegen requests use k = i % 5 vector paths; an old answer line
  without `vector` for trained_random is regenerated while trained / source_full lines are reused; rescore-only
  works on new lines.

### Task 2: KL window option (reference + train + eval)

**Files:** `src/persvectrain/cli.py`, `src/persvectrain/train.py`, `src/persvectrain/evaluate.py`,
`src/persvectrain/output.py`, `src/persvectrain/freegen.py` (reference suffix check accepts `_t{N}`), tests, spec
(Data).

- Produces: `cli.reference_path(out, model, tokens=512)`, `reference_meta_path(out, model, tokens=512)`;
  `reference --max-new-tokens N` (meta records `max_new_tokens`); `TrainConfig.kl_tokens = 512` (`--kl-tokens`);
  train / eval encode KL answers with `cfg.kl_tokens`; `train` checks the file exists before loading the model,
  message: `run: python -m persvectrain reference --model M --out O --max-new-tokens N`; train.json records
  `kl_tokens`; `config_from_train_json` defaults it to 512.
- Tests: path names (512 unchanged, 2048 suffixed); missing file message names `--max-new-tokens 2048`;
  `kl_tokens` truncates KL answers but not persona pairs; old train.json loads with 512; freegen accepts a `_t2048`
  reference.

### Task 3: `persona` command — vLLM answers + OpenRouter judge

**Files:** `src/persvectrain/persona.py` (new), `src/persvectrain/evaluate.py` (pass rule judge/coherence,
status), `src/persvectrain/cli.py` (`persona --out O --run-name R [--max-tokens 1024] [--judge-concurrency 32]
[--fresh]`; `train --no-judge`), `src/persvectrain/report.py` (pending also when persona_judge status is
"pending"; old runs without the key are not pending), `slurm/run.sbatch` (third step `persona` unless
`--no-judge`), tests, spec (new "Persona judge" section, Pass rule).

- Produces: `persona.questions(meta, src) -> list[(qid, text)]` (qid `"{slug}:{index}"`);
  `persona.request_plan(cfg, questions, slugs)`; `persona.run_persona(out, run_name, max_tokens=1024,
  open_engine=None, make_judges=None, fresh=False) -> dict | None` (engine and judges injectable for tests).
  Answers appended per chunk to `run_root/persona_answers.jsonl.gz` (freegen's read/write helpers, resume by key +
  seed + max_tokens + vector); judge scores appended to `run_root/persona_judged.jsonl.gz` keyed like the answers
  (resume: only unjudged answers are sent; a `None` score is retried on rerun). Trait name / description from
  PersVecFast `data/persona_vectors` (`load_traits`).
- eval.json: top-level `persona_judge = {"status": "pending" | "done" | "disabled", "judge_model",
  "max_tokens", "n_questions"}` (train writes pending, or disabled with `--no-judge`; `eval` keeps an existing
  block); per trait `persona_judge = {"unsteered": {...}, "elicited": {...}, kind: {c: {"qid", "score",
  "coherence", "score_mean", "score_se", "coherence_mean", "coherence_se", "n_tokens"}}}`; pass rule recomputed.
- Tests (fake engine, fake judge): request count = n_q x (2 + 3 x 2) per trait; seeds paired across kinds; system
  prompt neutral at +1 and eliciting at -1; random rotation; judge pass/fail on ties within 1 SE and beyond;
  coherence check; a JudgeError mid-run keeps answers and the rerun sends only unjudged ones; missing key fails
  before the engine opens; status pending -> done; old run without the key loads in report.

### Task 4: per-trait pick, ablation script, README

**Files:** `src/persvectrain/report.py`, `src/persvectrain/cli.py` (`report --per-trait [--write-profile P]`),
`slurm/ablate.sh` (new), `slurm/reeval.sbatch` (new), README, tests.

- Produces: `report.pick_per_trait(evals) -> {trait: run | None}`: among non-pending runs where the trait passes,
  the largest mean trained margin at c = +-1 (ties: first listed); `--write-profile P` writes
  `{trait: {"run", "vector": <the run's vllm_hook vector file>}}`.
- `slurm/reeval.sbatch --out O --run-name R`: `eval`, then `freegen`, then `persona` (vLLM env as run.sbatch).
- `slurm/ablate.sh <model> <layer>`: env `KL_TOKENS` (default "512 2048"), `LAMBDAS` (default "1 3 5 10"),
  `TRAITS`, `TRAIN_EXTRA` (default "--beta 0.01"), `SOURCE`, `OUT`. Run names `solo_kl{L}` for 512 (today's
  names) and `solo_kl{L}_t{N}` otherwise. Submits one `reference --max-new-tokens N` job per N != 512 whose file is
  missing; per grid cell: a run.sbatch job (dependency on its reference job) if `trained/<run>/train.json` is
  missing, else a reeval.sbatch job. Prints the report command with `--per-trait`.
- Tests: pick_per_trait on hand-made evals (pending skipped, tie order); ablate.sh dry run (`SBATCH=echo`-style
  stub as in tests/test_slurm.py) submits reeval for an existing run and run.sbatch for a missing one, with the
  reference dependency only for 2048.
- README: the ablation commands, `persona`, the judge key, `report --per-trait`.
