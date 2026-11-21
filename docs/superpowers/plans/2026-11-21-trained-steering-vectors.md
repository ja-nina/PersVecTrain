# Trained Steering Vectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one steering vector per trait at one layer (2B: L12) so the persona shows while reasoning stays
close to unsteered, alone (solo) and in mixes (joint), and say per run whether it worked.

**Architecture:** A frozen HF model with a forward hook that adds `c * v` to one decoder block's output at every
position. Only `v` is trained: BiPO's bidirectional persona loss on PersVecFast's kept D+/D- answers, plus a KL
term to the unsteered model on SuperGPQA 200-299 answers, plus (joint) an overlap penalty and 2-trait mix steps.
Output uses PersVecFast's file layout in its own run directory.

**Tech Stack:** Python >= 3.9, torch, transformers, `persvecfast` (editable install, same env), pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-trained-steering-vectors-design.md`

## The steps in plain words

1. Count usable pairs per trait on the cluster; drop traits with fewer than 20.
2. Per trait, split the questions 80/20 (seed 0); rebuild the start vector from the 80 % only.
3. Have the unsteered 2B answer SuperGPQA 200-299 once (first 512 tokens); save the answers.
4. Train: each step draws a sign d in {-1, +1}, 4 trait pairs and 4 SuperGPQA answers; loss = BiPO persona
   loss + lambda_kl * KL (+ overlap and mix steps in joint); only v moves. 20 epochs (about 160 steps).
5. Evaluate on the held-out 20 %: persona margin, KL, free-answer length, for trained / source / random twins;
   write pass or fail per trait.
6. Sweep lambda_kl in {0, 0.1, 1, 10} solo; then one joint run at the best lambda_kl.
7. Only if something passes: PersVecFast's judge + MMLU, then MA_Environments' lone-agent game (separate plans).

## Global Constraints

- Layer is an argument (2B: 12), never a constant; hook on `decoder_layers(model)[layer]` output = HF
  `hidden_states[layer + 1]` (PersVecFast smoke F5/F6). Log the layer's type from `config.layer_types` if present.
- Steering adds `c * v` at every position (prompt and answer). Train at c in {-1, +1}; mixes |w_i| <= 1.
- Settings: AdamW lr 5e-4, weight decay 0.05, cosine schedule, warmup 10 % of steps, batch 4 pairs + 4 KL
  prompts, beta 0.1, summed answer log-probabilities, 20 epochs, answers truncated to 512 tokens, seed 0.
- Prompts: `persvecfast.prompts.chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, question, enable_thinking=False)`.
- Never train on SuperGPQA 0-199 or 300-339, MMLU, or HLE. Never write into the source dir or an existing run dir.
- Every run writes `train.json` (full config, seed, PersVecTrain + PersVecFast git commits, source paths),
  `steps.jsonl` (append-only, one line per step), `eval.json`.
- Every reported vector has a norm-matched random twin (`persvecfast.extract.random_control`).
- Tests run without a GPU, on a tiny random Llama model built from a config (no download).
- Model kept frozen (`requires_grad_(False)` on all parameters); v is fp32, cast to the model dtype when added.

## Review Focus

1. Answer token boundary: the log-probability must cover exactly the answer's tokens (after the chat template's
   assistant header), truncated to 512. Test in Task 3.
2. `_raw.pt` row k is question `kept_questions[k]`, not question k: the train-split start vector must pick rows by
   that list. Test in Task 2.
3. A decoder layer may return a tensor or a tuple; the hook must handle both and leave no hook behind after an
   exception. Test in Task 3.
4. Re-running with an existing `--run-name`, or `--out` pointing into the source dir, must stop before training.
   Test in Task 7.
5. A trait below 20 kept pairs, or with no `_responses.json`, is dropped with a printed reason, not a crash.
   Test in Task 2.

## File Structure

```
PersVecTrain/
  pyproject.toml, README.md, .gitignore          (Task 1)
  data/supergpqa_200_299.jsonl                    (Task 1: copied from MA_Environments)
  src/persvectrain/
    data.py        pairs, split, start vector, SuperGPQA prompts, pair counts      (Task 2)
    steering.py    steer() hook, encode(), answer log-probs                         (Task 3)
    losses.py      persona, KL, overlap losses, persona margin                      (Task 4)
    train.py       TrainConfig, train() solo + joint                                (Task 5, 6)
    evaluate.py    held-out eval, free generation, pass rule                        (Task 8)
    output.py      run dir layout, writing vectors and logs                         (Task 7)
    cli.py, __main__.py   `check`, `reference`, `train`, `report` subcommands       (Task 7, 9)
  slurm/train.sbatch, slurm/sweep.sh                                                (Task 9)
  tests/conftest.py (tiny model + char tokenizer), tests/test_*.py
```

---

### Task 1: Repo skeleton

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`, `src/persvectrain/__init__.py`, `tests/conftest.py`,
  `tests/test_conftest.py`, `data/supergpqa_200_299.jsonl`

**Interfaces:**
- Produces: fixtures `tiny_model` (LlamaForCausalLM: vocab 128, hidden 16, 3 layers, 2 heads, intermediate 32,
  seed 0, frozen, eval mode) and `char_tok` (`CharTokenizer`: `apply_chat_template(messages, tokenize=False,
  add_generation_prompt=True, **kw) -> str` as `"<s>{system}<u>{user}<a>"`, `__call__(text,
  add_special_tokens=False) -> {"input_ids": [ord(c) % 128 ...]}`, `decode(ids) -> str`, `eos_token_id = 0`).

- [ ] **Step 1:** `pyproject.toml` like PersVecFast's (setuptools, src layout, `name = "persvectrain"`,
  dependencies `torch`, `transformers`; `persvecfast` documented in README as an editable install, not a pip
  dependency; pytest `pythonpath = ["src", "."]`). `.gitignore`: `out/`, `__pycache__/`, `*.egg-info`,
  `slurm-logs/`, `.cache/`, `hle*`.
- [ ] **Step 2:** Copy lines 201-300 (questions 200-299, 0-based) of
  `MA_Environments/testbed/data/supergpqa.jsonl` into `data/supergpqa_200_299.jsonl`. Verify:
  `wc -l` = 100 and the first id equals line 201's id.
- [ ] **Step 3:** Write `tests/test_conftest.py::test_tiny_model_runs`: `tiny_model(torch.tensor([[1, 2, 3]]))`
  returns logits of shape `[1, 3, 128]`, and `output_hidden_states=True` gives 4 hidden states.
- [ ] **Step 4:** `pip install -e ../PersVecFast -e .[test]` then `pytest -q`. Expected: 1 passed.
- [ ] **Step 5:** Commit `"skeleton: package, tiny test model, SuperGPQA 200-299"`.

### Task 2: Data (pairs, split, start vector, prompts, counts)

**Files:** Create `src/persvectrain/data.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `@dataclass Pair: index: int; question: str; pos: str; neg: str` (index = question index in the trait file)
  - `load_pairs(src: Path, slug: str) -> list[Pair]`: pairs where both sides have `kept == True`.
  - `split(pairs: list, seed: int = 0, frac: float = 0.8) -> tuple[list, list]`: shuffle the sorted indices with
    `random.Random(seed)`, first `round(frac * n)` are train. Same seed + same pairs = same split.
  - `start_vector(src: Path, slug: str, layer: int, train_indices: list[int]) -> torch.Tensor` (fp32): mean over
    train rows of `_raw.pt["positive"][layer]["residual"]` minus the same for negative, rows chosen through
    `_raw.pt["kept_questions"]`.
  - `full_source_vector(src, slug, layer) -> torch.Tensor`: `torch.load(src / f"{slug}.pt")[layer]`.
  - `supergpqa_prompts(path: Path) -> list[dict]` with keys `id`, `question` (text), `answer` (letter). Text =
    `question + "\n\n" + "\n".join(f"{L}. {opt}") + "\n\nWhich option is correct?"` (the careful-reasoning format).
  - `usable_traits(src: Path, slugs: list[str], min_pairs: int = 20) -> tuple[list[str], dict[str, str]]`:
    kept slugs, and a reason per dropped slug (`"missing"` or `"N kept pairs < 20"`).

- [ ] **Step 1: Tests** (write fake `{slug}_responses.json` / `_raw.pt` / `.pt` into `tmp_path`):
  - `test_load_pairs_only_kept`: 3 questions, question 1 has `kept: False` -> indices `[0, 2]`.
  - `test_split_deterministic_and_disjoint`: 40 pairs -> 32/8, disjoint, identical on a second call, different
    with seed 1.
  - `test_start_vector_uses_kept_questions`: `kept_questions = [0, 2, 5]`, rows with known values; train indices
    `[2, 5]` -> mean of rows 1 and 2 only.
  - `test_supergpqa_format`: one record with options `["x", "y"]` -> text ends `"A. x\nB. y\n\nWhich option is
    correct?"`; and the real file has 100 records.
  - `test_usable_traits_drops_with_reason`: one slug with 25 kept pairs, one with 5, one missing ->
    kept `[first]`, reasons `{"second": "5 kept pairs < 20", "third": "missing"}`.
- [ ] **Step 2:** Run `pytest tests/test_data.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `data.py` with the signatures above.
- [ ] **Step 4:** Run `pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `"data: kept pairs, 80/20 split, train-split start vector, SuperGPQA prompts"`.

### Task 3: Steering hook and answer log-probabilities

**Files:** Create `src/persvectrain/steering.py`, `tests/test_steering.py`

**Interfaces:**
- Consumes: `persvecfast.smoke.decoder_layers`, `persvecfast.prompts.chat_prompt, NEUTRAL_SYSTEM_PROMPT`.
- Produces:
  - `steer(model, layer: int, vec: torch.Tensor | None)` context manager: adds `vec` (already scaled by c; None
    = no-op) to the block output at all positions; tensor or tuple output; hook removed on exit and on exception.
  - `encode(tok, question: str, answer: str, max_answer_tokens: int = 512) -> tuple[torch.Tensor, int]`:
    `(ids [1, T], start)`; prompt and answer tokenised separately and concatenated, so `ids[0, start:]` are the
    answer's first <= 512 tokens.
  - `answer_logprob(model, ids, start) -> torch.Tensor` (scalar, summed log p of answer tokens).
  - `answer_logprobs(model, ids, start) -> torch.Tensor` (`[T_answer, vocab]` log-softmax, fp32) for the KL.

- [ ] **Step 1: Tests** (tiny model, char tokenizer):
  - `test_steer_shifts_hidden_state`: with `steer(m, 1, v)`, `hidden_states[2]` (layer 1's output) equals the
    unsteered one plus `v` at every position (atol 1e-5), and `hidden_states[1]` is unchanged.
  - `test_steer_none_and_zero_are_noops`: logits equal (atol 0) to unsteered.
  - `test_hook_removed_after_exception`: raise inside the context; afterwards logits equal unsteered.
  - `test_encode_answer_boundary`: `tok.decode(ids[0, start:])` equals the answer; a 600-char answer keeps 512.
  - `test_answer_logprob_matches_manual`: equals the sum of `log_softmax(logits[0, t-1])[ids[0, t]]` for
    `t >= start`.
  - `test_tuple_output_layer`: wrap one layer so it returns `(tensor,)`; steering still shifts the output.
- [ ] **Step 2:** Run `pytest tests/test_steering.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement; assert `vec.shape == (hidden,)` and `ids.shape[0] == 1` (fail loudly).
- [ ] **Step 4:** Run `pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `"steering: block-output hook at all positions, answer log-probabilities"`.

### Task 4: Losses

**Files:** Create `src/persvectrain/losses.py`, `tests/test_losses.py`

**Interfaces:**
- Produces:
  - `persona_margin(lp_c_pos, lp_0_pos, lp_c_neg, lp_0_neg, sign: int) -> Tensor` =
    `sign * ((lp_c_pos - lp_0_pos) - (lp_c_neg - lp_0_neg))`
  - `persona_loss(..., sign, beta: float = 0.1) -> Tensor` = `-logsigmoid(beta * persona_margin(...))`
  - `kl_loss(logp_0: Tensor, logp_c: Tensor) -> Tensor` = mean over rows of `sum p_0 (logp_0 - logp_c)`
  - `overlap_loss(vecs: Tensor) -> Tensor` = sum over i < j of `cos(v_i, v_j) ** 2`; 0 for one vector.

- [ ] **Step 1: Tests:** margin 0 -> loss `log 2`; sign flips the margin; KL of equal inputs = 0 and > 0
  otherwise; overlap of orthogonal vectors 0, of `[v, 2v]` = 1, of one vector 0.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run, expect PASS.
- [ ] **Step 5:** Commit `"losses: BiPO persona loss and margin, KL to unsteered, overlap penalty"`.

### Task 5: Solo training loop

**Files:** Create `src/persvectrain/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces:
  - `@dataclass TrainConfig`: `model: str, layer: int, mode: str ("solo"|"joint"), traits: list[str],
    source: Path, out: Path, run_name: str, lambda_kl: float = 1.0, lambda_o: float = 0.0, epochs: int = 20,
    lr: float = 5e-4, weight_decay: float = 0.05, warmup_frac: float = 0.1, beta: float = 0.1, batch: int = 4,
    kl_batch: int = 4, max_answer_tokens: int = 512, init: str = "dom" ("dom"|"zero"), mix_every: int = 2,
    seed: int = 0`
  - `@dataclass TraitData: slug: str; train: list[Pair]; held: list[Pair]; start: Tensor; source: Tensor`
  - `@dataclass RefAnswer: id: str; question: str; answer_text: str; gold: str`
  - `train(model, tok, cfg: TrainConfig, traits: dict[str, TraitData], kl_train: list[RefAnswer],
    log_step: Callable[[dict], None]) -> dict[str, Tensor]`

- [ ] **Step 1: Tests** (tiny model; synthetic pairs: pos answers all `"aaaa..."`, neg all `"bbbb..."`):
  - `test_solo_raises_train_margin`: 3 epochs, lr 1e-2, lambda_kl 0 -> mean train persona margin at both signs
    after > before.
  - `test_kl_term_keeps_kl_lower`: same seed, lambda_kl 0 vs 100 -> KL on `kl_train` at c=+1 is lower with 100.
  - `test_only_vector_changes`: model parameters bit-identical after training.
  - `test_log_step_fields`: every logged dict has `step, trait, sign, loss_p, loss_kl, norm, lr`; steps =
    `epochs * ceil(len(train) / batch)` per trait.
  - `test_zero_init`: `init="zero"` starts at norm 0 (first logged norm before the update is 0).
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement. Per step: draw `d` from `{-1, +1}` (seeded `torch.Generator`); for each of the
  `batch` pairs and `kl_batch` reference answers, one sequence at a time: unsteered pass under `torch.no_grad()`,
  steered pass with `steer(model, layer, d * v)`, loss / (batch + kl_batch), `backward()`; then one optimiser step.
  Unsteered log-probabilities of the training pairs are computed once at the start and reused. Cosine schedule via
  `transformers.get_cosine_schedule_with_warmup`. KL reference answers cycle through `kl_train`.
- [ ] **Step 4:** Run `pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `"train: solo BiPO + KL loop on a frozen model, one vector per trait"`.

### Task 6: Joint mode (overlap + mix steps)

**Files:** Modify `src/persvectrain/train.py`; extend `tests/test_train.py`

**Interfaces:**
- Produces: the same `train(...)`; with `mode="joint"` all trait vectors are one optimiser's parameters; every
  `mix_every`-th step is a mix step; logged dicts gain `kind ("single"|"mix"), weights, loss_o`.

- [ ] **Step 1: Tests:**
  - `test_joint_overlap_reduces_cosine`: two traits with start vectors at cosine 0.9; lambda_o 10, lambda_kl 0
    -> final |cos| < 0.9.
  - `test_mix_step_reference_is_other_traits`: a mix step of traits A, B with weights (wA, wB): the reference
    log-probabilities for A's pairs are taken under `steer(wB * vB)`, the steered ones under
    `steer(wA * vA + wB * vB)` (check by patching `answer_logprob` to record the vector it ran under).
  - `test_mix_weights_bounded`: all logged weights in [-1, 1], nonzero, 2 traits per mix step.
  - `test_solo_equals_joint_without_extras`: one trait, joint with lambda_o 0 and `mix_every = 0` gives the same
    vector as solo (same seed).
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement. Mix step: draw 2 distinct traits and weights `|w| ~ U[0.25, 1]`, random signs; persona
  loss per trait with `sign(w_i)`, on that trait's pairs, reference = steered by the other trait only; KL under the
  full mix against unsteered. `L_o` added every step with lambda_o.
- [ ] **Step 4:** Run `pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `"train: joint mode, overlap penalty and 2-trait mix steps"`.

### Task 7: Output layout and the `check`, `reference`, `train` commands

**Files:** Create `src/persvectrain/output.py`, `src/persvectrain/cli.py`, `src/persvectrain/__main__.py`,
`tests/test_output.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 2, 5, 6; `persvecfast.models.MODELS`, `persvecfast.smoke.load_hf`,
  `persvecfast.extract.random_control`, `persvecfast.util.write_json, save_pt, read_json`.
- Produces:
  - `run_dir(out: Path, run_name: str, model: str) -> Path` = `out/trained/<run_name>/persona/<model>/bf16`
  - `prepare_run(cfg) -> Path`: raises `FileExistsError` if `out/trained/<run_name>` exists, `ValueError` if the run
    dir is inside `cfg.source`.
  - `write_vectors(dir, slug, layer, trained: Tensor, src: Path, run_name: str) -> None`: copies
    `{slug}_responses.json`, `{slug}_raw.pt`, `{slug}.json` (+ `trained` block: run name, layer, source norm,
    trained norm), writes `{slug}.pt` (source dict with `[layer]` replaced) last, `vllm_hook/{slug}_L{layer}.pt`
    `{"dir": v}` and `_random.pt` (`random_control(v, seed + layer)`).
  - `python -m persvectrain check --model M --source S --traits ...`: prints kept pairs per trait and drops.
  - `python -m persvectrain reference --model M --out O`: the unsteered model answers `data/supergpqa_200_299.jsonl`
    once (model card sampling from `MODELS[M]`, `max_new_tokens 512`, seed per question via
    `persvecfast.models.request_seed`), writes `O/trained/reference/<model>.jsonl`; skips if present.
  - `python -m persvectrain train ...` (flags = TrainConfig fields, the spec's interface): checks, splits
    (SuperGPQA reference 80/20 with seed), trains, writes vectors, `train.json`, `steps.jsonl`, then runs Task 8's
    eval.

- [ ] **Step 1: Tests:**
  - `test_prepare_run_refuses_existing_and_source`: both errors raised; nothing created.
  - `test_write_vectors_layout`: files exist; `{slug}.pt[layer]` equals trained, other layers equal the source;
    `vllm_hook` file is `{"dir": tensor}`; random twin has the same norm; source files untouched (hash).
  - `test_train_json_has_config_seed_commits`: keys `config, seed, persvectrain_commit, persvecfast_commit,
    sources, layer_type, dropped_traits`.
  - `test_cli_train_end_to_end_tiny`: with `load_hf` and `MODELS` patched to the tiny model + char tokenizer and a
    fake source dir, `main(["train", ...])` writes the full layout.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run `pytest -q`, expect PASS.
- [ ] **Step 5:** Commit `"output layout and check / reference / train commands"`.

### Task 8: Built-in evaluation and pass rule

**Files:** Create `src/persvectrain/evaluate.py`, `tests/test_evaluate.py`; modify `cli.py` (train calls it)

**Interfaces:**
- Produces:
  - `evaluate(model, tok, cfg, traits: dict[str, TraitData], trained: dict[str, Tensor], kl_held:
    list[RefAnswer]) -> dict` with, per trait, per vector kind (`trained`, `source_full`, `source_train`,
    `source_at_trained_norm`, `trained_random`, `source_random`) and per c in `[-1, -0.5, 0.5, 1]`:
    `margin_mean, margin_se, kl_mean, kl_se`; plus `norms`, `cos_with_source`; joint: `cos_matrix` and
    `mix_margin` vs `alone_margin` on held-out pairs with weights (+-1, +-1).
  - `free_generation(model, tok, vec, prompts: list[RefAnswer], seed) -> dict`: `mean_tokens, cut_share,
    accuracy, accuracy_se`; 4096 new tokens, model card sampling except presence penalty (HF has none; logged),
    answer = last match of `r"\b([A-J])\b"` after `"answer"` (case-insensitive), else wrong. Run for c = +-1 on
    `trained`, `source_full`, `trained_random`, and once unsteered.
  - `pass_rule(trait_eval: dict) -> dict`: for c in (+1, -1): `persona` (trained margin >= source_full margin -
    1 SE of the difference), `kl` (trained KL < source_full KL, or tie within 1 SE counts as fail),
    `length` (|trained tokens - unsteered| <= |source tokens - unsteered|), `random` (trained KL <= trained_random
    KL + 1 SE and same for length); `passed = all`.
  - Writes `eval.json`; prints one table (rows trait x kind x c).

- [ ] **Step 1: Tests:** pure `pass_rule` cases: all better -> passed; persona lower by more than 1 SE -> fails on
  `persona`; within 1 SE -> persona passes; KL worse than random twin -> fails on `random`; each failing check is
  named. `evaluate` on the tiny model returns every kind x c key and SEs (n >= 2); `free_generation` with a fake
  generate returning `"... answer is C"` scores accuracy 1 for gold C.
- [ ] **Step 2:** Run, expect FAIL. **Step 3:** Implement. **Step 4:** Run `pytest -q`, expect PASS.
- [ ] **Step 5:** Commit `"eval: held-out margin, KL, free-answer length, random twins, pass rule"`.

### Task 9: Slurm, report, cluster runs

**Files:** Create `slurm/train.sbatch`, `slurm/sweep.sh`, `src/persvectrain/report.py`, `tests/test_report.py`

**Interfaces:**
- Produces:
  - `slurm/train.sbatch` like PersVecFast's `run_model.sbatch` (python from the active env, `HF_HOME` off home),
    `#SBATCH -p gpu-l40 --gres=gpu:1 -c 10 --mem=64G --time=12:00:00`, runs `python -m persvectrain "$@"`.
  - `slurm/sweep.sh <model> <layer>`: one `reference` job, then 4 `train --mode solo` jobs
    (`--lambda-kl 0 / 0.1 / 1 / 10`, run names `solo_kl<value>`) with `--dependency=afterok`.
  - `python -m persvectrain report --out O --runs solo_kl0 solo_kl0.1 ...`: one table of pass/fail per run and
    trait; picks the run that passes the most traits (tie: larger mean trained margin at c = +-1), or prints
    "no run passes".

- [ ] **Step 1: Tests:** `test_report_picks_most_passing` and `test_report_none_passes` on fake `eval.json` files.
- [ ] **Step 2-4:** FAIL, implement, PASS.
- [ ] **Step 5:** Commit `"slurm sweep and report"`; push (after the GitHub repo exists, see below).
- [ ] **Step 6 (cluster, the user runs):** clone next to MASteering, `pip install -e /path/to/PersVecFast -e .`,
  then `python -m persvectrain check --model qwen3.5-2b --source /scratch/inf0/user/nzukowsk/PersVecFast/out/persona/qwen3.5-2b/bf16 --traits Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness`.
- [ ] **Step 7:** One short timing job: `train --epochs 1` on one trait; read seconds per step from `steps.jsonl`.
  If over ~10 s per step, stop and report before the sweep.
- [ ] **Step 8:** `bash slurm/sweep.sh qwen3.5-2b 12`; then `report`; then one `--mode joint --lambda-o 1` run at
  the picked lambda_kl.

## After this plan (not built here)

Stage 2 (PersVecFast `steer judge,summarize capability` on a trained dir) and stage 3 (MA_Environments lone-agent
SuperGPQA with `vectors_dir`, `traits_dir`, fixed `layer: 12`) get their own short plans once a run passes.
