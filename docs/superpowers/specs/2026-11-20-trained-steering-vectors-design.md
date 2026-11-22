# Trained steering vectors (PersVecTrain)

2026-11-20. Status: design approved 2026-11-21, not built. Separate repo `PersVecTrain` (the user's call),
next to PersVecFast; it reads PersVecFast's outputs and imports `persvecfast` for model settings and loading.

## Why

The thesis steers LLM agents in multi-agent games (MA_Environments) with PersVecFast persona vectors. The goal is
an agent **profile**: numbers per trait, e.g. `{assertive: 0.5, empathy: -0.3}`, for many agents at once. Each
agent's persona should show, and its task performance should stay where it is unsteered.

Today's vectors are a difference of means (D+ answers minus D- answers, per layer). On the 2B (Qwen3.5-2B, layer
12, SuperGPQA hard, a lone agent, 200 questions, MA_Environments `debate_fix` / `debate_l12`):
- the strength that clearly moves the trait also costs accuracy (l12 c = +1: assertive -0.205, sycophantic -0.14,
  opportunistic -0.13; unsteered 0.325, SE about 0.037);
- much of that cost is the push's size: a random direction of the same norm often costs as much;
- prompt-only steering (prefill), removing a "careful reasoning" component, and steering to a target level did not
  beat plain steering at a smaller c: prefill moved no persona at all, the other two traded persona for accuracy
  at the same rate as a smaller c;
- answer length moves a lot with steering (assertive +1: 7.5k chars, -1: 43k, unsteered 19k).

Nothing in a difference of means asks the vector to leave reasoning alone. Training can: a persona objective plus a
term that keeps the steered model close to the unsteered one on reasoning prompts (BiPO / RePS for the first,
KL-then-steer for the second). For profiles, the trait vectors must also add up: a mix should show each of its
traits, not double-count or cancel them (MAT-Steer).

## Goals

1. A repo `PersVecTrain` (package `persvectrain`, installed next to an editable `persvecfast` in the same env) that trains one steering vector per trait at one layer, starting from the
   difference-of-means vector, with
   - a persona loss: steering +c makes the trait's D+ answers more likely than its D- answers, -c the reverse;
   - a keep-reasoning loss: KL between the unsteered and steered next-token distributions on reasoning prompts;
2. Two modes, one codebase:
   - **solo** (approach 1): each trait on its own (persona + KL);
   - **joint** (approach 2): all traits of a run together, plus an overlap penalty between trait vectors and
     training steps that steer with random mixes of traits (persona per trait and KL under the mix).
   Solo is joint with the overlap and mix weights at zero.
3. Output that MA_Environments and PersVecFast's own steer / judge / capability stages load unchanged
   (same file layout), in its own directory.
4. A built-in held-out evaluation at the end of each run, so a run says by itself whether it worked.

## Non-goals (for now)

- Text profiles / a hypernetwork from profiles to vectors (HyperSteer). Numbers per trait only.
- Low-rank adapters (ReFT) or LoRA: both need engine changes or slow batched serving.
- Several layers per vector, or any change to vLLM-Hook.
- Thinking-mode models (`--thinking`): the 2B in non-thinking mode first.

## Anti-goals (would be a failure even if it runs)

- A vector that "keeps reasoning" by shrinking until the persona is gone. Every result reports persona and
  reasoning side by side, plus the trained norm next to the source norm.
- Training on anything the games evaluate on: SuperGPQA questions 0-199 (debate_acc) and 300-339
  (careful-reasoning trait) stay out; MMLU stays out (capability stage). HLE never.
- Overwriting `out/persona/...` or any existing vector, or changing PersVecFast.
- Results that cannot be traced to their exact config and seed.

## Constraints

- 2B first, one GPU (L40 / A40, 48 GB), HF transformers + torch, PyTorch hooks on the decoder block output (the
  convention PersVecFast's capture and vLLM-Hook steering share: captured layer l = HF hidden_states[l + 1], checked
  in smoke F5/F6). Qwen3.5 is hybrid (Gated DeltaNet + full attention): residual-stream steering at a layer
  boundary is architecture-agnostic, so no layer-type restriction; log the layer's type anyway.
- Resolve num_hidden_layers / hidden_size from the config; the layer is an argument (2B: 12, the dense sweep's
  pick), never a constant in code.
- Use strength at play time: |c| <= 1 (the user steers only within that). Train at c in {-1, +1} (solo) and mixes
  with |w_i| <= 1 (joint).
- Each run writes its full config, seed, PersVecFast git commit, source vector paths, and per-step losses
  (append-only JSONL) next to its outputs.
- Every trained vector is reported with a norm-matched random control (MA_Environments' `random_control`
  convention: random direction, same norm).
- Tests run without a GPU (tiny random model or a fake), as the rest of PersVecFast's tests.

## Data

- Persona loss: `out/persona/<model>/bf16/{trait}_responses.json`: `positive` / `negative` lists of
  `{question, system_prompt, response, kept, ...}`. Only `kept` pairs (both sides non-empty, not refused), the
  same pairs the vector was built from. The training prompt is the question **without** the system prompt (the
  vector replaces the instruction). D+ may come from the abliterated fallback model (`extraction_model`); the loss
  still scores it under the base model. Split the kept questions 80/20 by question (fixed seed) into train / held-out.
  *(Call made after review.)* The starting vector is rebuilt from the **train split only** (difference of means of
  `{trait}_raw.pt` over the train pairs at the layer), so the held-out pairs are unseen by both the start and the
  trained vector; the built-in eval compares against this train-split source as well as the full one. Held-out is
  small (about 8 pairs per trait of 40): report SEs, and treat stage 3 (the games) as the clean test. Stage 2 judges
  all 40 questions (mostly training questions), so its trait scores are read as "does it still steer", not as a
  held-out result.
- First build step: check on the cluster that each trait has enough kept pairs (and was built from the
  non-thinking model, not a `nothink/` window); drop traits with fewer than about 20 kept pairs and say which.
  Answers are truncated to their first 512 tokens for training (most are cut at 2048; the opening carries the persona).
- Keep-reasoning loss (updated 2026-11-21: vLLM free generation, larger pool): a pool of 260 SuperGPQA questions, records 200-299 and 340-499 (0-based) of
  MA_Environments' `testbed/data/supergpqa.jsonl`, in this repo as `data/supergpqa_pool.jsonl`, formatted like the
  careful-reasoning trait (question, lettered options, "Which option is correct?"). Split with seed 0 into 160
  KL-train / 100 held-out (was 100 questions, 80/20). Records 0-199 (the games' questions) and 300-339
  (careful-reasoning) stay out. The unsteered model's own answers are generated once (HF, model card sampling) and
  cached as `<model>_pool260.jsonl`; the KL is over those answer tokens (teacher forcing), first 512 tokens.
  KL window (2026-11-22): `reference --max-new-tokens N` makes `<model>_pool260_t{N}.jsonl` (N = 512 keeps the name
  above); `TrainConfig.kl_tokens` (`--kl-tokens`, default 512) picks that file and cuts the KL answers to N tokens in
  train and eval, while persona pairs stay at `max_answer_tokens`. train.json records `kl_tokens`; older ones read as 512.
  Risk: if the games sample questions past 199 they overlap this pool; check before reading game results as
  independent of the KL questions.

## Losses

For one trait with vector v at layer l, steering adds c * v to the block output at every position (prompt and
answer, as vLLM-Hook add_vector with phase both). logp_c(y | x) is the answer's summed log-probability with
steering c; logp_0 without.

- Persona (BiPO, bidirectional): for c in {+1, -1}, with s = sign(c),
  `L_p = -log sigmoid(beta * s * ([logp_c(y+) - logp_0(y+)] - [logp_c(y-) - logp_0(y-)]))`
  averaged over training pairs and both signs. Per-token normalisation of the log-probabilities is the builder's
  call (report which).
- Keep reasoning: `L_kl = mean over answer tokens of KL(p_0 || p_c)` on the SuperGPQA prompts, both signs.
- Joint only:
  - overlap: `L_o = sum over trait pairs i < j of cos(v_i, v_j)^2`;
  - mix steps: draw a mix of 2 traits with weights w_i, |w_i| <= 1 (signs random), steer with sum_i w_i v_i;
    persona loss for each trait in the mix with sign(w_i) (its own questions, the other traits' vectors present),
    and L_kl under the mix. *(Call made after review.)* The reference in a mix step's persona loss is the model
    steered by the **other** traits of the mix only (not the unsteered model), so each vector is trained on its own
    contribution inside the mix; the KL reference stays the unsteered model.
- Total: `L = L_p + lambda_kl * L_kl + lambda_o * L_o (+ mix-step terms, same weights)`.

lambda_kl is the knob between persona and reasoning. The first run sweeps it (0 = plain BiPO, the "is the KL term
what helps?" ablation, then e.g. 0.1, 1, 10); its value is not fixed in this design.

## Evaluation

1. **Built-in, at the end of every run** (held-out splits, no API calls): per trait, for the trained vector, the
   source (difference-of-means) vector, and each one's random control, at c = +-0.5 and +-1:
   - persona margin: the held-out mean of `s * ([logp_c(y+) - logp_0(y+)] - [logp_c(y-) - logp_0(y-)])`;
   - KL on held-out SuperGPQA prompts;
   - vector norm and cosine with the source vector; in joint runs, the cosine matrix between trait vectors and the
     persona margin of each trait inside held-out 2-trait mixes vs alone;
   - *(calls made after review)* the source vector also **rescaled to the trained vector's norm**, so "trained
     vs source" is compared at equal push size as well as at equal c (push size alone explains much of the cost);
     and a free-generation check, since teacher-forced KL may not track what steering does to free answers
     (updated 2026-11-21: vLLM free generation, larger pool): answers to the 100 held-out SuperGPQA questions with vLLM (PersVecFast's steer engine; the model card's
     sampling, up to 15360 new tokens, the games' "ANSWER: <letter>" instruction), their mean length and accuracy next
     to unsteered. Run as a separate `freegen` step after `train`.
   *(Updated 2026-11-22: 5 random directions.)* The random control is a set of 5 norm-matched random directions,
   seeds `RANDOM_CONTROL_SEED + layer + 1000 k`, k = 0..4 (k = 0 is the single twin used before); `trained_random`
   and `source_random` keep their names. Teacher-forced, their margin and KL per item are the mean over the 5
   directions at c = +-1 (the pass-rule c values); at c = +-0.5 only k = 0 is used (cost). The 5 per-direction KL
   (and margin) means are stored in `kinds[...][c]["per_direction"]` (`per_direction_margin`), eval.json has
   `"random_dirs": 5`.
   Written as one JSON per run plus a printed table.

   **Pass rule** (the user's choice: the strict version). A trait's trained vector passes when, on held-out data
   at c = +1 and at c = -1:
   - it moves the persona **at least as much** as the source vector (persona margin >= source);
   - it changes reasoning **less** than the source at the same c: lower KL, and free-generation length closer
     to unsteered, and (accuracy@c, (updated 2026-11-21: vLLM free generation, larger pool)) **at least as accurate** as `source_full`, paired by question, a
     difference within 1 SE counts as a tie;
   - its reasoning cost (KL, length change, accuracy change) is **no worse than its own random twin's** (random
     direction, same norm as the trained vector; from 2026-11-22 the twin's KL is the mean over 5 random directions
     and its free-answer length comes from the rotation below; the form of the check is unchanged).
   - *(Updated 2026-11-22: persona judge.)* Judged persona answers (`persona_judge`, "Persona judge" below), paired
     by question, ties within 1 SE: at c = +1 the trained vector's judge expression score is **at least** the
     source's (judge@+1); at c = -1 (under the eliciting prompt) it **suppresses at least as much** (judge@-1,
     score_source - score_trained >= -SE); and its judge coherence is at least the source's (coherence@c). A question
     with a None judge score in either kind is left out of the pair. Without judge results (status pending or
     disabled) these checks are skipped and listed in the trait's `pass.skipped`.
   *(Calls made by me.)* Held-out is small, so "at least as much" / "no worse" count a difference within 1 SE as a
   tie. A run's eval.json says pass / fail per trait and per check. Among the swept lambda_kl values, the one that
   passes for the most traits moves on to stages 2 and 3 (ties: the larger mean persona margin). If none passes for
   any trait, stop and report; no stages 2-3.
2. **PersVecFast's own stages** on the trained output dir: `steer judge,summarize capability` (judge trait scores
   and coherence, MMLU), comparable with the dense-sweep results at L12. If the pipeline needs an option to steer
   from a different vectors dir than it extracted into, add one (the builder's call how; `copy_vectors` already
   exists for the thinking case).
3. **The games** (MA_Environments, separate, after 1 and 2 look right): the lone-agent SuperGPQA setup with
   `vectors_dir` pointing at the trained dir, single traits at c = +-0.5 / +-1 and a few 2-trait mixes, against the
   source vectors at the same c and the random controls; accuracy and the monitor's trait score.

## Output layout

`<out>/trained/<run_name>/` with
- `persona/<model>/bf16/{trait}.pt`: the source per-layer dict with the trained layer replaced (MA_Environments'
  VectorStore indexes by layer, and the target / project_out options need the other files below);
- `persona/<model>/bf16/{trait}_responses.json`, `{trait}_raw.pt`, `{trait}.json`: copied from the source (the
  `.json` gains a `trained` block: run name, layer, source norm, trained norm);
- `persona/<model>/bf16/vllm_hook/{trait}_L{layer}.pt`: `{"dir": vector}`;
- `train.json` (config, seed, commit, sources), `steps.jsonl` (losses per step), `eval.json` (section 1).

So MA_Environments uses it with `vectors_dir: <out>/trained/<run_name>/persona/<model>/bf16`, plus
`traits_dir: /scratch/inf0/user/nzukowsk/PersVecFast/data/persona_vectors` (VectorStore's default looks three levels
up for `data/`, which the trained layout does not have) and a fixed `layer:` (not `auto`: the copied `.json`'s
layer scores belong to the source vector). *(Call made after review.)*

## Interface

    python -m persvectrain train --model qwen3.5-2b --layer 12 --mode solo|joint \
        --traits Assertive Sycophantic Opportunistic Empathy Evil Sarcastic Agreeableness \
        --source /scratch/inf0/user/nzukowsk/PersVecFast/out --out /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --run-name <name> --lambda-kl 1 [--lambda-o 1] [--steps N] [--seed 0]

plus a `slurm/train.sbatch` wrapper like the existing ones. Resumable is nice to have, not required (runs are
expected to take minutes to an hour per trait set).

## Approaches considered

1. Solo: each trait trained alone (persona + KL). Simplest, parallel, but nothing makes vectors combine.
2. Joint: all traits together with an overlap penalty and mix steps (MAT-Steer-like). Trains composition directly.
3. Low-rank adapter per trait (ReFT): more expressive, but needs a vLLM-Hook patch and mixes less cleanly.

Picked: **1 and 2**, built as one codebase (solo = joint with lambda_o = 0 and no mix steps). 3 dropped.

## Settings from the papers

BiPO (2406.00045): AdamW, learning rate 5e-4, weight decay 0.05, cosine schedule with warmup, batch 4, beta 0.1,
the sign d drawn from {-1, +1} once per step, the vector added at every token position, whole-answer
log-probabilities (summed, the DPO convention). BiPO starts from zero; here the start is the train-split
difference of means (the design's choice), and a zero start is a CLI option for comparison. Epochs: BiPO uses 5-20
on 300-700 pairs; here about 32 pairs, so 20 epochs = 160 steps per trait (warmup 10 % of steps).
KL-then-steer (2406.15518) trains a rank-128 LoRA so the model tolerates steering (KL of steered vs unsteered on
benign prompts); here the same KL is put on the vector instead. That combination is not in either paper.

## Free-generation check (updated 2026-11-21: vLLM free generation, larger pool)

The 2B's unsteered SuperGPQA answers are long (about 19k characters; at 4096 tokens 75 % were cut in
target_clamp_pilot.py), so the check no longer truncates: vLLM through PersVecFast's steer engine, up to 15360 new
tokens (the games' limit), the model card's sampling, the games' "ANSWER: <letter>" instruction, on the 100 held-out
questions. Per trait: `trained`, `source_full` and `trained_random` at c = +1 and -1, plus one unsteered run; each
steered request uses the same seed for every kind (paired). An answer is the last `ANSWER: X`; none counts as wrong.
Per answer: tokens, whether it was cut, correct. With 100 questions the accuracy SE is about 0.05, so accuracy is back
in the pass rule (accuracy@c above) next to persona margin, KL and length. Until `freegen` has run, a run's free-gen
status is "pending" and the report shows it as not passed.

*(Updated 2026-11-22: 5 random directions.)* `trained_random` answers rotate over the 5 random directions: the
question at position i of the held list (train.json's order) uses direction k = i % 5, vector files
`freegen_vectors/{slug}_trained_random_k{k}.pt` (no plain `{slug}_trained_random.pt`). Every saved answer line of a
steered request carries the vector file name; a saved `trained_random` line is reused on resume only when its name is
the expected one (older lines without it are regenerated). The per-trait free-gen block of `trained_random` lists the
direction of each item (`"direction"`).

## Persona judge (added 2026-11-22)

`persona --out O --run-name R` answers the run's held persona questions on vLLM and scores them with PersVecFast's
OpenRouter judge (`persvecfast.judge`, expression model `openai/gpt-oss-120b`, temperature 0; key `OPENROUTER_API_KEY`
from the environment or PersVecFast's `.env`, checked before the engine opens). Questions: the union of every trait's
held pair questions (train.json splits), by trait (config order) then pair index, qid `"{slug}:{index}"`; the same
list for every trait. Per trait and question: `unsteered` (neutral system prompt), `elicited` (PersVecFast's
`positive_instruction(trait, False)` as system prompt, no steering), and `trained`, `source_full`, `trained_random` at
c = +1 under the neutral prompt and at c = -1 under the eliciting prompt (PersVecFast's suppression setting).
Thinking off, up to 1024 new tokens (`--max-tokens`). Seeds `request_seed(seed, "persona", c, qi)` for steered
requests (paired across kinds and traits), `"neutral"` / `"elicited"` instead of c unsteered. `trained_random` rotates
the 5 directions by question position (k = i % 5, as in `freegen`). The judge gives a trait-expression score and a
coherence score (0-100); an empty answer gets 0 / 0 without a call; a None score (unparseable reply) is excluded
from means and paired differences and retried on rerun.

Answers are appended per engine chunk to `persona_answers.jsonl.gz` (the freegen line format), scores per judge batch
to `persona_judged.jsonl.gz` (lines: key, seed, score, coherence, plus max_tokens and vector; the last line per key
wins). Both resume. A judge error retrying cannot fix (no credit, bad key) stops the job with a message and keeps both
files; a rerun sends only the unjudged answers.

eval.json: top-level `persona_judge = {"status": "pending" | "done" | "disabled", "judge_model", "max_tokens",
"n_questions", ...}` (`train` writes pending, or disabled with `--no-judge`; `eval` keeps an existing block; a run
without the key is not pending); per trait `persona_judge = {"unsteered": B, "elicited": B, kind: {"c=+1": B,
"c=-1": B}}` with B = per-question lists `qid`, `score`, `coherence`, `n_tokens` and `score_mean` / `score_se` /
`coherence_mean` / `coherence_se`. The pass rule's judge and coherence checks (above) are recomputed.

## Left to the builder

Optimiser and learning rate, batch composition, steps / epochs, whether the vector's norm is free or bounded
(report it either way), beta, per-token vs summed log-probabilities, how the unsteered reference log-probabilities
are cached, the exact mix-sampling scheme, the code layout inside `train/`.
