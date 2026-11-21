# Free generation on vLLM, larger question pool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The end-of-run check answers 100 held-out SuperGPQA questions with free answers up to 15360 tokens,
on vLLM, and the pass rule checks accuracy again.

**Architecture:** Training and the teacher-forced check stay in HF (`train`). A new `freegen` command, a separate
process in the same Slurm job, loads PersVecFast's vLLM-Hook steering engine
(`persvecfast.mia.open_engine(spec, "steer")`), generates every (trait, vector kind, c) plus unsteered in one
batched call, and completes `eval.json`. The SuperGPQA pool grows from 100 to 260 questions.

**Tech Stack:** as before, plus vLLM-Hook through `persvecfast.mia` (cluster only; tests use a fake engine).

**Spec:** `docs/superpowers/specs/2026-11-20-trained-steering-vectors-design.md` (Task 1 updates its Data,
Evaluation and Free-generation sections to this plan).

## Why (timing run, 2026-11-21)

HF free generation: 7 batches x 20 answers at 4096 tokens took 21 min, 75 % of unsteered answers were cut at
4096 and accuracy was 0.1 +- 0.07. The user asked for longer answers and a larger question pool.

## Global Constraints

- Pool: MA_Environments `testbed/data/supergpqa.jsonl` records 200-299 and 340-499 (0-based), 260 questions.
  0-199 (games) and 300-339 (careful-reasoning) stay out. Split with seed 0: 160 KL-train, 100 held.
- Free answers: up to 15360 new tokens (the games' `max_tokens`), the model card sampling from
  `MODELS[model]` (vLLM applies presence_penalty), prompt = the held question text + "\n\nEnd your reply with
  'ANSWER: <letter>', one of A, B, ...". (the games' instruction), chat template with the neutral system prompt and
  thinking off. Answer = last `ANSWER: X` (letters of that question's options, case-insensitive, not followed by a
  letter); none = wrong. Cut = vLLM `finish_reason == "length"`.
- Requests: unsteered once; per trait `trained`, `source_full`, `trained_random` at c = +1 and -1. Seed per
  request `request_seed(cfg.seed, "freegen", c, question_id)` for steered (the same for every kind and trait:
  paired), `request_seed(cfg.seed, "freegen", "unsteered", question_id)` unsteered.
- Steering in vLLM: `{"layer": cfg.layer, "coeff": c, "vector_path": <file {"dir": vector}>}`. The three vectors per
  trait are written to `run_root/freegen_vectors/{slug}_{kind}.pt` (self-contained; the random twin is
  `random_control(trained, 0 + layer)` as everywhere else).
- Pass rule gains `accuracy@c` per c in (+1, -1): mean(correct_trained - correct_source_full) >= -SE (paired, a tie
  within 1 SE passes, "at least as accurate"). Length checks stay (now on uncensored-ish lengths). Random check
  unchanged (user's strict rule; decision on loosening it is open).
- `eval.json`: `train` writes it with `free_gen.status = "pending"` (or "disabled" with `--no-free-gen`), no HF free
  generation any more; `freegen` fills per trait x kind x c and unsteered: per-item `tokens`, `cut`, `correct`,
  `answer`, plus means/SE, recomputes the pass rule, sets `status = "done"`. `report` treats "pending" as not passed
  (shown PENDING, never picked).
- Never write into the source dir; vLLM steer engine needs the env vars PersVecFast's `slurm/run_model.sbatch` sets
  (`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_CACHE_ROOT`, `MIA_APERTURE_DIR`,
  `PVF_MIA_DIR` per job, cleaned on exit).

## Review Focus

1. Pairing: the same question id gets the same seed and the same position in every kind's per-item list.
2. A trait dropped by `train` or a run with `--no-free-gen` must not crash `freegen` (skip with a message).
3. Re-running `freegen` on a run whose status is "done" overwrites the free-gen block only, never vectors/train.json.
4. Reference answers: the old 100-question file must not be silently used for the 260 pool.
5. Batching: thousands of requests go in chunks (`MODELS[model].batch_size`), not one giant call.

---

### Task 1: 260-question pool, 160/100 split, reference file, spec

**Files:** `data/supergpqa_pool.jsonl` (new; 260 records in source order: 200-299 then 340-499),
remove `data/supergpqa_200_299.jsonl`; `src/persvectrain/cli.py` (reference + split), tests, spec.

- Produces: `reference_path(out, model) -> out/trained/reference/<model>_pool260.jsonl` (+ `.meta.json`, which
  records `pool: "supergpqa 200-299 + 340-499"`, n 260); `split_reference(refs, seed, n_held=100)` -> (160, 100),
  seeded like `data.split`; `train` errors clearly if the reference file is missing. Reference generation stays
  HF at 512 tokens (only the first 512 tokens are used).
- Tests: pool has 260 records, ids equal source records 200-299 + 340-499, no overlap with 0-199 / 300-339;
  split sizes 160/100, deterministic; reference path name; missing file message.
- Spec: update "Data" (pool, split), "Evaluation" item 1 free-generation bullet and "Free-generation check"
  section (vLLM, 15360 tokens, 100 held, accuracy back in the pass rule), "Pass rule" (accuracy@c).

### Task 2: `freegen` on vLLM, accuracy in the pass rule

**Files:** `src/persvectrain/freegen.py` (new), `src/persvectrain/evaluate.py` (drop HF free generation; pass rule
accuracy; status handling), `src/persvectrain/cli.py` (`freegen --out O --run-name R [--max-tokens 15360]`),
tests.

- Produces: `build_requests(cfg, run, held: list[RefAnswer], tok) -> list[Request]` (keys
  `(slug | "-", kind, c, question_id)`), `parse_answer(text, n_options) -> str | None`,
  `run_freegen(out, run_name, max_tokens, open_engine=persvecfast.mia.open_engine) -> dict` (engine injectable
  for tests), result folding into eval.json and the pass rule.
- Tests (fake engine returning canned texts/finish reasons): request count = 100 x (1 + 6 x n_traits) in the real
  config shape (use small n in tests); identical seeds across kinds for one (c, question); vector files written
  with the right tensors; parse cases (`"... ANSWER: C"`, lowercase, `"ANSWER: (D)"`, last one wins, a letter
  beyond the options = None, no marker = None); accuracy check passes on a tie within 1 SE and fails beyond;
  status pending -> done; `--no-free-gen` run is skipped with a message; rerun overwrites only eval.json.

### Task 3: Slurm job, sweep options, report PENDING, README

**Files:** `slurm/run.sbatch` (new: `train "$@"` then `freegen --out ... --run-name ...` in one job, with the vLLM
env vars and per-job dirs from PersVecFast's run_model.sbatch, `-p gpu-l40 --gres=gpu:1 -c 10 --mem=64G
--time=20:00:00`), `slurm/sweep.sh` (uses run.sbatch; `TRAIN_EXTRA` env forwarded to every train), 
`src/persvectrain/report.py` (PENDING), README (commands), tests for report.

- run.sbatch takes the train arguments; it reads `--out` and `--run-name` from them for the freegen step (a small
  parse in bash, or `python -m persvectrain freegen --from-train-args "$@"` — implementer's call, tested).
- README: install/update, reference, a timing job (one trait, 1 epoch, freegen on), the sweep with
  `TRAIN_EXTRA="--beta 0.01"`, report, joint run.
