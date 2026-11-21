"""`freegen`: the free-generation check on vLLM (spec, "Free-generation check").

Answers to the run's held-out SuperGPQA questions with PersVecFast's vLLM-Hook steer engine: once unsteered, and
per trait `trained`, `source_full` and `trained_random` at c = +1 and -1, up to `max_tokens` new tokens with the
model card's sampling (vLLM applies presence_penalty). Every steered request for one (c, question) has the same seed
whatever the kind and trait, so answers are paired by question. Results (per answer: tokens, cut, answer,
correct) go into the run's eval.json, which gets its pass rule recomputed and free_gen.status "done". Only the
tokenizer is loaded here (chat template); vLLM loads the model.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import torch
from persvecfast.engine import Request, Result
from persvecfast.extract import RANDOM_CONTROL_SEED, random_control, vector_file
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import chunks, log, read_json, save_pt, write_json

from persvectrain.data import LETTERS, full_source_vector, supergpqa_prompts
from persvectrain.evaluate import (FREE_GEN_C, FREE_GEN_KINDS, apply_pass, c_key, fg_cols, mean_se, pass_lines)
from persvectrain.output import REPO, config_from_train_json, load_reference, run_dir, run_root, source_dir
from persvectrain.train import RefAnswer, TrainConfig

POOL = REPO / "data" / "supergpqa_pool.jsonl"
MAX_TOKENS = 15360  # the games' max_tokens
REFERENCE_SUFFIX = "_pool260.jsonl"  # cli.reference_path: the reference answers of the 260-question pool
UNSTEERED = "unsteered"
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")


def instruction(n_options: int) -> str:
    """The games' answer instruction for a question with n_options options."""
    return "\n\nEnd your reply with 'ANSWER: <letter>', one of " + ", ".join(LETTERS[:n_options]) + "."


def parse_answer(text: str, n_options: int) -> str | None:
    """The last `ANSWER` (any case), optional `:` / `*` / spaces / `(`, then one of the question's option letters
    (any case, not followed by another letter); upper-case, or None when there is none."""
    letters = LETTERS[:n_options]
    matches = re.findall(rf"ANSWER[\s:*(]*([{letters}])(?![A-Z])", text, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else None


def load_tokenizer(model_key: str):
    """The model's tokenizer only (for the chat template); never the HF model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELS[model_key].hf_repo)


def default_open_engine():
    """PersVecFast's vLLM-Hook engine factory; imported here so vllm is only needed on the cluster."""
    from persvecfast.mia import open_engine

    return open_engine


# ---------------------------------------------------------------- inputs


def pool_options(pool: Path) -> dict[str, int]:
    """Number of options per question id in the pool file."""
    out = {}
    for line in Path(pool).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = len(r["options"])
    return out


def held_questions(meta: dict, pool: Path) -> tuple[list[RefAnswer], dict[str, int]]:
    """The run's held reference questions in train.json's order and their option counts.

    Refuses a reference that is not the 260-question pool's file, and held ids or question texts / golds that
    differ from the pool (an old or rebuilt reference would otherwise be used silently).
    """
    ref_file = Path(meta["sources"]["reference"])
    if not ref_file.name.endswith(REFERENCE_SUFFIX):
        raise SystemExit(f"freegen: the run's reference {ref_file} is not the 260-question pool's "
                         f"(*{REFERENCE_SUFFIX}); retrain on the pool")
    refs = {r.id: r for r in load_reference(ref_file)}
    texts = {it["id"]: it for it in supergpqa_prompts(Path(pool))}
    n_options = pool_options(pool)
    held = []
    for qid in meta["splits"]["reference"]["held"]:
        if qid not in refs or qid not in texts:
            raise SystemExit(f"freegen: held question {qid} is missing from {ref_file} or the pool {pool}")
        r = refs[qid]
        if r.question != texts[qid]["question"] or r.gold != texts[qid]["answer"]:
            raise SystemExit(f"freegen: held question {qid} in {ref_file} differs from the pool {pool}")
        held.append(r)
    return held, {r.id: n_options[r.id] for r in held}


def write_freegen_vectors(cfg: TrainConfig, slugs: list[str]) -> dict[str, dict[str, Path]]:
    """run_root/freegen_vectors/{slug}_{kind}.pt = {"dir": vector} for the three free-gen kinds; absolute paths.

    trained: the run's vllm_hook file as saved; source_full: the source's vector at the layer; trained_random:
    random_control(trained, RANDOM_CONTROL_SEED + layer), as write_vectors and evaluate.vector_kinds.
    """
    d = (run_root(cfg.out, cfg.run_name) / "freegen_vectors").resolve()
    vec_dir = run_dir(cfg.out, cfg.run_name, cfg.model)
    out: dict[str, dict[str, Path]] = {}
    for slug in slugs:
        path = vector_file(vec_dir, slug, cfg.layer)
        if not path.exists():
            raise SystemExit(f"freegen: trained vector {path} missing")
        trained = torch.load(path)["dir"]
        vecs = {"trained": trained, "source_full": full_source_vector(source_dir(cfg), slug, cfg.layer),
                "trained_random": random_control(trained, RANDOM_CONTROL_SEED + cfg.layer)}
        out[slug] = {}
        for kind in FREE_GEN_KINDS:
            out[slug][kind] = d / f"{slug}_{kind}.pt"
            save_pt(out[slug][kind], {"dir": vecs[kind].detach().cpu().clone()})
    return out


def build_requests(cfg: TrainConfig, held: list[RefAnswer], n_options: dict[str, int],
                   vector_paths: dict[str, dict[str, Path]], tok, max_tokens: int = MAX_TOKENS) -> list[Request]:
    """Unsteered once per held question, then per trait x kind x c; keys (slug | "-", kind, c, question id).

    Seeds: request_seed(cfg.seed, "freegen", c, qid) for every steered kind and trait (paired),
    request_seed(cfg.seed, "freegen", "unsteered", qid) unsteered.
    """
    prompts = {r.id: chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, r.question + instruction(n_options[r.id]), False)
               for r in held}
    reqs = [Request(("-", UNSTEERED, 0.0, r.id), prompts[r.id],
                    request_seed(cfg.seed, "freegen", UNSTEERED, r.id), None, max_tokens) for r in held]
    for slug, paths in vector_paths.items():
        for kind in FREE_GEN_KINDS:
            for c in FREE_GEN_C:
                steer = {"layer": cfg.layer, "coeff": c, "vector_path": str(paths[kind])}
                reqs += [Request((slug, kind, c, r.id), prompts[r.id], request_seed(cfg.seed, "freegen", c, r.id),
                                 dict(steer), max_tokens) for r in held]
    return reqs


# ---------------------------------------------------------------- scoring


def score(results: list[Result], held: list[RefAnswer], n_options: dict[str, int]) -> dict:
    """One free-gen block: per-answer lists in held order plus means (cut = vLLM finish_reason "length")."""
    answers = [parse_answer(r.text, n_options[q.id]) for r, q in zip(results, held)]
    correct = [a is not None and a == q.gold.upper() for a, q in zip(answers, held)]
    cut = [r.finish_reason == "length" for r in results]
    tokens = [int(r.n_tokens) for r in results]
    acc, acc_se = mean_se([float(x) for x in correct])
    return {"n": len(held), "mean_tokens": mean_se([float(t) for t in tokens])[0],
            "cut_share": mean_se([float(x) for x in cut])[0], "accuracy": acc, "accuracy_se": acc_se,
            "qid": [q.id for q in held], "tokens": tokens, "cut": cut, "answer": answers, "correct": correct}


def fold(results: dict[tuple, Result], held: list[RefAnswer], n_options: dict[str, int],
         slugs: list[str]) -> tuple[dict, dict[str, dict]]:
    """(unsteered block, {slug: {"unsteered": ..., kind: {"c=+1": ..., "c=-1": ...}}})."""
    def block(slug: str, kind: str, c: float) -> dict:
        return score([results[(slug, kind, c, q.id)] for q in held], held, n_options)

    unsteered = block("-", UNSTEERED, 0.0)
    per_trait = {s: {"unsteered": unsteered,
                     **{kind: {c_key(c): block(s, kind, c) for c in FREE_GEN_C} for kind in FREE_GEN_KINDS}}
                 for s in slugs}
    return unsteered, per_trait


# ---------------------------------------------------------------- run


def run_freegen(out: Path, run_name: str, max_tokens: int = MAX_TOKENS, open_engine=None,
                pool: Path = POOL) -> dict | None:
    """Free generation for one finished run; returns its updated eval.json, or None (with a message) when the run
    has no train.json / eval.json, did not finish, was trained with --no-free-gen, or has no trait to answer.

    Writes run_root/freegen_vectors/ and the free-gen parts of eval.json only (top-level free_gen, per trait
    free_gen and pass, summary, timing.free_gen_s); train.json and the run's vectors are never touched.
    """
    root = run_root(Path(out), run_name)
    for name in ("train.json", "eval.json"):
        if not (root / name).exists():
            print(f"freegen: {root / name} missing; nothing to do (run `train` first)", flush=True)
            return None
    meta = read_json(root / "train.json")
    if not meta.get("finished"):
        print(f"freegen: run {root} did not finish training (train.json finished false); nothing to do",
              flush=True)
        return None
    status = read_json(root / "eval.json").get("free_gen", {}).get("status")
    if status == "disabled":
        print(f"freegen: run {root} was evaluated with --no-free-gen (free_gen status disabled); nothing to do",
              flush=True)
        return None
    cfg = config_from_train_json(meta, out)
    for slug, reason in (meta.get("dropped_traits") or {}).items():
        print(f"freegen: trait {slug} was dropped by train ({reason}); skipped", flush=True)
    ev_traits = read_json(root / "eval.json")["traits"]
    slugs = [s for s in cfg.traits if s in ev_traits]
    for s in cfg.traits:
        if s not in slugs:
            print(f"freegen: trait {s} has no teacher-forced eval in eval.json; skipped", flush=True)
    if not slugs:
        print(f"freegen: no trait of {root} to answer; nothing to do", flush=True)
        return None
    if status == "done":
        log(f"freegen: {root} already has free-gen results; rerun overwrites them")

    spec = MODELS[cfg.model]
    held, n_options = held_questions(meta, pool)
    tok = load_tokenizer(cfg.model)
    paths = write_freegen_vectors(cfg, slugs)
    reqs = build_requests(cfg, held, n_options, paths, tok, max_tokens)
    open_engine = open_engine or default_open_engine()
    log(f"freegen {run_name}: {len(reqs)} requests ({len(held)} questions, traits {slugs}), {max_tokens} new tokens, "
        f"chunks of {spec.batch_size}")
    t0 = time.monotonic()
    results: dict[tuple, Result] = {}
    with open_engine(spec, "steer") as engine:
        for chunk in chunks(reqs, spec.batch_size):
            for r in engine.generate(chunk):
                # only what scoring needs: thousands of 15k-token answers' token ids would fill the RAM
                results[r.key] = Result(r.key, r.text, r.n_tokens, r.finish_reason)
            log(f"freegen {run_name}: {len(results)}/{len(reqs)} answers")
    missing = [r.key for r in reqs if r.key not in results]
    if missing:
        raise RuntimeError(f"freegen: the engine returned no answer for {len(missing)} requests, e.g. {missing[0]}")
    unsteered, per_trait = fold(results, held, n_options, slugs)

    # re-read: eval.json may have been rewritten (`eval`) while vLLM was generating
    ev = read_json(root / "eval.json")
    ev["free_gen"] = {"enabled": True, "status": "done", "engine": "vllm-hook (persvecfast.mia steer)",
                      "max_tokens": max_tokens,
                      "sampling": {k: v for k, v in asdict(spec).items() if k in SAMPLING_FIELDS},
                      "presence_penalty_applied": True, "batch_size": spec.batch_size, "n_held": len(held),
                      "instruction": "End your reply with 'ANSWER: <letter>', one of A, B, ...",
                      "vectors": {s: {k: str(p) for k, p in kv.items()} for s, kv in paths.items()},
                      "unsteered": unsteered}
    for s in slugs:
        ev["traits"][s]["free_gen"] = per_trait[s]
    apply_pass(ev, [s for s in ev["traits"]])
    ev.setdefault("timing", {})["free_gen_s"] = round(time.monotonic() - t0, 2)
    write_json(root / "eval.json", ev)
    log(f"freegen {run_name}: done in {ev['timing']['free_gen_s']} s; wrote {root / 'eval.json'}")
    print_freegen_table(ev)
    return ev


def print_freegen_table(ev: dict) -> None:
    """tokens / cut / acc per trait x kind x c and unsteered, then the pass line per trait."""
    lines = [f"{'trait':<18} {'kind':<16} {'c':>4} {'tokens':>8} {'cut':>5} {'acc':>12}"]
    lines.append(f"{'-':<18} {UNSTEERED:<16} {'0':>4} {fg_cols(ev['free_gen'].get('unsteered'))}")
    for s, t in ev["traits"].items():
        for kind in FREE_GEN_KINDS:
            for c in FREE_GEN_C:
                k = c_key(c)
                lines.append(f"{s:<18} {kind:<16} {k[2:]:>4} {fg_cols(t.get('free_gen', {}).get(kind, {}).get(k))}")
    lines += pass_lines(ev)
    print("\n".join(lines), flush=True)
