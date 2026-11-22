"""`freegen`: the free-generation check on vLLM (spec, "Free-generation check").

Answers to the run's held-out SuperGPQA questions with PersVecFast's vLLM-Hook steer engine: once unsteered, and
per trait `trained`, `source_full` and `trained_random` at c = +1 and -1, up to `max_tokens` new tokens with the
model card's sampling (vLLM applies presence_penalty). Every steered request for one (c, question) has the same seed
whatever the kind and trait, so answers are paired by question. Results (per answer: tokens, cut, answer,
correct) go into the run's eval.json, which gets its pass rule recomputed and free_gen.status "done". Only the
tokenizer is loaded here (chat template); vLLM loads the model.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import torch
from persvecfast.engine import Request, Result
from persvecfast.extract import vector_file
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import chunks, log, read_json, save_pt, write_json

from persvectrain.data import LETTERS, full_source_vector, supergpqa_prompts
from persvectrain.evaluate import (FREE_GEN_C, FREE_GEN_KINDS, RANDOM_DIRS, apply_pass, c_key, fg_cols, mean_se,
                                   pass_lines, random_set)
from persvectrain.output import REPO, config_from_train_json, load_reference, run_dir, run_root, source_dir
from persvectrain.train import RefAnswer, TrainConfig

POOL = REPO / "data" / "supergpqa_pool.jsonl"
MAX_TOKENS = 15360  # the games' max_tokens
REFERENCE_SUFFIX = "_pool260.jsonl"  # the 260-question pool's reference answers (cli.reference_path)
REFERENCE_NAME_RE = re.compile(r"_pool260(_t\d+)?\.jsonl$")  # a longer / shorter KL window adds _t{N}
UNSTEERED = "unsteered"
ANSWERS_FILE = "freegen_answers.jsonl.gz"  # in run_root: every saved answer, one gzip member per chunk
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")


def instruction(n_options: int) -> str:
    """The games' answer instruction for a question with n_options options."""
    return "\n\nEnd your reply with 'ANSWER: <letter>', one of " + ", ".join(LETTERS[:n_options]) + "."


def parse_answer(text: str, n_options: int) -> str | None:
    """The answer letter as MA_Environments' games parse it (QADebateParser with testbed.parsers.base.keyword_value,
    reimplemented here, not imported): the LAST match, case-insensitive, of `ANSWER`, optional `*` / `_`, a required
    `:` or `=`, optional opening `*`, `_`, `[`, `(`, `<`, quotes or backticks, then one of the question's option letters not followed by
    another letter. Upper-cased; None when nothing matches (none = wrong). A later marker without a valid letter
    is no match, so an earlier valid one wins, as in the games."""
    letters = LETTERS[:n_options]
    pattern = rf"ANSWER\s*[*_]*\s*[:=]\s*[*_\[(<\"'`]*\s*([{letters}{letters.lower()}](?![A-Za-z]))"
    matches = list(re.finditer(pattern, text or "", re.IGNORECASE))
    return matches[-1].group(1).upper() if matches else None


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
    if not REFERENCE_NAME_RE.search(ref_file.name):
        raise SystemExit(f"freegen: the run's reference {ref_file} is not the 260-question pool's "
                         f"(*{REFERENCE_SUFFIX} or *_pool260_t<N>.jsonl); retrain on the pool")
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


def random_name(slug: str, k: int) -> str:
    """File name of the slug's k-th random direction (k = 0..4)."""
    return f"{slug}_trained_random_k{k}.pt"


def freegen_vector_paths(cfg: TrainConfig, slugs: list[str]) -> dict[str, dict[str, Path | list[Path]]]:
    """{slug: {"trained": path, "source_full": path, "trained_random": [5 paths, k = 0..4]}} under
    run_root/freegen_vectors, absolute; nothing written."""
    d = (run_root(cfg.out, cfg.run_name) / "freegen_vectors").resolve()
    return {slug: {"trained": d / f"{slug}_trained.pt", "source_full": d / f"{slug}_source_full.pt",
                   "trained_random": [d / random_name(slug, k) for k in range(RANDOM_DIRS)]} for slug in slugs}


def _trained_and_source(cfg: TrainConfig, slugs: list[str]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    vec_dir = run_dir(cfg.out, cfg.run_name, cfg.model)
    out = {}
    for slug in slugs:
        path = vector_file(vec_dir, slug, cfg.layer)
        if not path.exists():
            raise SystemExit(f"freegen: trained vector {path} missing")
        out[slug] = (torch.load(path)["dir"], full_source_vector(source_dir(cfg), slug, cfg.layer))
    return out


def freegen_vectors(cfg: TrainConfig, slugs: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    """{slug: {kind: vector}} for the three free-gen kinds; nothing written.

    trained: the run's vllm_hook file as saved; source_full: the source's vector at the layer; trained_random: the
    single twin random_control(trained, RANDOM_CONTROL_SEED + layer) = k = 0 of the random set (kldepth uses it;
    write_freegen_vectors writes all 5 directions).
    """
    return {slug: {"trained": t, "source_full": sf, "trained_random": random_set(t, cfg.layer)[0]}
            for slug, (t, sf) in _trained_and_source(cfg, slugs).items()}


def write_freegen_vectors(cfg: TrainConfig, slugs: list[str]) -> dict[str, dict[str, Path | list[Path]]]:
    """run_root/freegen_vectors/{slug}_trained.pt, _source_full.pt and _trained_random_k{0..4}.pt =
    {"dir": vector}; absolute paths (freegen_vector_paths). No plain {slug}_trained_random.pt."""
    out = freegen_vector_paths(cfg, slugs)
    for slug, (t, sf) in _trained_and_source(cfg, slugs).items():
        save_pt(out[slug]["trained"], {"dir": t.detach().cpu().clone()})
        save_pt(out[slug]["source_full"], {"dir": sf.detach().cpu().clone()})
        for k, r in enumerate(random_set(t, cfg.layer)):
            save_pt(out[slug]["trained_random"][k], {"dir": r.detach().cpu().clone()})
    return out


NORM_MATCHED = "source_at_trained_norm"  # source_full scaled to |trained|: the length control (bench, persona)


def write_norm_matched(cfg: TrainConfig, slugs: list[str], paths: dict[str, dict]) -> dict[str, dict]:
    """Adds run_root/freegen_vectors/{slug}_source_at_trained_norm.pt = source_full * |trained| / |source_full| to
    paths (in place) and returns it. Not one of freegen's own kinds."""
    d = (run_root(cfg.out, cfg.run_name) / "freegen_vectors").resolve()
    for slug, (t, sf) in _trained_and_source(cfg, slugs).items():
        p = d / f"{slug}_{NORM_MATCHED}.pt"
        save_pt(p, {"dir": (sf.float() * (t.float().norm() / sf.float().norm())).cpu().clone()})
        paths[slug][NORM_MATCHED] = p
    return paths


def vector_for(paths: dict[str, dict], slug: str, kind: str, pos: int) -> Path:
    """The vector file of a request for the question at held position pos: trained_random rotates, k = pos % 5."""
    p = paths[slug][kind]
    return p[pos % RANDOM_DIRS] if isinstance(p, list) else p


def request_plan(cfg: TrainConfig, held: list[RefAnswer], slugs: list[str],
                 kinds: tuple[str, ...] = FREE_GEN_KINDS) -> list[tuple[tuple, int]]:
    """(key, seed) of every free-gen request in order: unsteered once per held question, then per trait x kind x c;
    keys (slug | "-", kind, c, question id). trained_random's direction is not in the key: held position i uses
    random direction i % 5 (vector_for); saved answers carry the file name.

    Seeds: request_seed(cfg.seed, "freegen", c, qid) for every steered kind and trait (paired),
    request_seed(cfg.seed, "freegen", "unsteered", qid) unsteered.
    """
    plan = [(("-", UNSTEERED, 0.0, r.id), request_seed(cfg.seed, "freegen", UNSTEERED, r.id)) for r in held]
    for slug in slugs:
        for kind in kinds:
            for c in FREE_GEN_C:
                plan += [((slug, kind, c, r.id), request_seed(cfg.seed, "freegen", c, r.id)) for r in held]
    return plan


def build_requests(cfg: TrainConfig, held: list[RefAnswer], n_options: dict[str, int],
                   vector_paths: dict[str, dict], tok, max_tokens: int = MAX_TOKENS,
                   kinds: tuple[str, ...] = FREE_GEN_KINDS) -> list[Request]:
    """The requests of request_plan (traits = vector_paths' keys) with their prompts and steering."""
    prompts = {r.id: chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, r.question + instruction(n_options[r.id]), False)
               for r in held}
    pos = {r.id: i for i, r in enumerate(held)}
    reqs = []
    for key, seed in request_plan(cfg, held, list(vector_paths), kinds):
        slug, kind, c, qid = key
        steer = None if kind == UNSTEERED else \
            {"layer": cfg.layer, "coeff": c, "vector_path": str(vector_for(vector_paths, slug, kind, pos[qid]))}
        reqs.append(Request(key, prompts[qid], seed, steer, max_tokens))
    return reqs


# ---------------------------------------------------------------- saved answers (run_root/freegen_answers.jsonl.gz)


ANSWER_FIELDS = ("key", "text", "n_tokens", "finish_reason", "max_tokens", "seed")


def _gunzip_tolerant(data: bytes) -> tuple[bytes, bool]:
    """Decompress concatenated gzip members (one per appended chunk); (bytes, damaged). A cut or corrupt member
    ends the read: what it decompressed so far is kept (its last line may be partial, the reader drops it)."""
    out = []
    while data:
        d = zlib.decompressobj(wbits=31)
        try:
            out.append(d.decompress(data))
        except zlib.error:
            return b"".join(out), True
        if not d.eof:
            return b"".join(out), True
        data = d.unused_data
    return b"".join(out), False


def read_answers(path: Path) -> tuple[list[dict], bool]:
    """The saved answer lines in file order, and whether the file was damaged (cut gzip member, a line that is
    not a complete JSON answer); damaged lines are dropped."""
    if not path.exists():
        return [], False
    raw, damaged = _gunzip_tolerant(path.read_bytes())
    lines = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            damaged = True
            continue
        if not isinstance(d, dict) or any(k not in d for k in ANSWER_FIELDS):
            damaged = True
            continue
        lines.append(d)
    return lines, damaged


def write_answers(path: Path, lines: list[dict], append: bool = True) -> None:
    """Append (one gzip member, closed on return) or rewrite the file with these answer lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        with gzip.open(path, "at", encoding="utf-8") as f:
            f.write("".join(json.dumps(ln) + "\n" for ln in lines))
        return
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(ln) + "\n" for ln in lines))
    os.replace(tmp, path)


def answer_line(r: Result, req: Request) -> dict:
    line = {"key": list(req.key), "text": r.text, "n_tokens": int(r.n_tokens), "finish_reason": r.finish_reason,
            "max_tokens": req.max_tokens, "seed": req.seed}
    if req.steer:
        line["vector"] = Path(req.steer["vector_path"]).name  # which random direction trained_random used
    return line


def expected_vectors(plan: list[tuple[tuple, int]], held: list[RefAnswer]) -> dict[tuple, str]:
    """The vector file name a saved trained_random answer must carry: the question's rotated direction."""
    pos = {r.id: i for i, r in enumerate(held)}
    return {key: random_name(key[0], pos[key[3]] % RANDOM_DIRS) for key, _ in plan if key[1] == "trained_random"}


def saved_results(lines: list[dict], plan: list[tuple[tuple, int]], max_tokens: int,
                  vectors: dict[tuple, str] | None = None) -> dict[tuple, Result]:
    """Results of the plan's requests found in the saved lines (same key, seed and max_tokens; the last line wins).

    vectors {key: file name} (expected_vectors): a line of such a key is reused only when its "vector" equals the
    name, so an answer made with another random direction (or an old line without "vector") is regenerated.
    """
    seeds = dict(plan)
    vectors = vectors or {}
    out = {}
    for ln in lines:
        key = tuple(ln["key"])
        if key in seeds and ln["seed"] == seeds[key] and ln["max_tokens"] == max_tokens \
                and (key not in vectors or ln.get("vector") == vectors[key]):
            out[key] = Result(key, ln["text"], ln["n_tokens"], ln["finish_reason"])
    return out


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
         slugs: list[str], kinds: tuple[str, ...] = FREE_GEN_KINDS) -> tuple[dict, dict[str, dict]]:
    """(unsteered block, {slug: {"unsteered": ..., kind: {"c=+1": ..., "c=-1": ...}}})."""
    def block(slug: str, kind: str, c: float) -> dict:
        b = score([results[(slug, kind, c, q.id)] for q in held], held, n_options)
        if kind == "trained_random":
            b["direction"] = [i % RANDOM_DIRS for i in range(len(held))]  # random direction k of each item
        return b

    unsteered = block("-", UNSTEERED, 0.0)
    per_trait = {s: {"unsteered": unsteered,
                     **{kind: {c_key(c): block(s, kind, c) for c in FREE_GEN_C} for kind in kinds}}
                 for s in slugs}
    return unsteered, per_trait


# ---------------------------------------------------------------- run


def run_freegen(out: Path, run_name: str, max_tokens: int | None = None, open_engine=None,
                pool: Path = POOL, fresh: bool = False, rescore_only: bool = False) -> dict | None:
    """Free generation for one finished run; returns its updated eval.json, or None (with a message) when the run
    has no train.json / eval.json, did not finish, was trained with --no-free-gen, or has no trait to answer.

    Every engine chunk's answers are appended to run_root/freegen_answers.jsonl.gz (one gzip member per chunk) as
    soon as it is done; a rerun reuses the saved answers of the same requests (key, seed and max_tokens) and
    generates only the rest (resume), `fresh` deletes the file first. Scoring always reads the file.
    `rescore_only`: no engine and no tokenizer, re-parse the saved answers (e.g. after a parser change); every
    request must have a saved answer; max_tokens None = the cap eval.json records (else MAX_TOKENS).

    Writes run_root/freegen_vectors/, the answers file and the free-gen parts of eval.json only (top-level
    free_gen, per trait free_gen and pass, summary, timing.free_gen_s; rescore keeps timing.free_gen_s);
    train.json and the run's vectors are never touched.
    """
    if fresh and rescore_only:
        raise ValueError("freegen: fresh and rescore_only exclude each other")
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
    answers_path = root / ANSWERS_FILE
    if rescore_only:
        if not answers_path.exists():
            print(f"freegen: {answers_path} missing; nothing to rescore (run `freegen` first)", flush=True)
            return None
        if max_tokens is None:
            max_tokens = read_json(root / "eval.json").get("free_gen", {}).get("max_tokens") or MAX_TOKENS
    else:
        max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
        if status == "done":
            log(f"freegen: {root} already has free-gen results; rerun overwrites them "
                f"({'regenerating' if fresh else 'saved answers reused'})")
        if fresh and answers_path.exists():
            log(f"freegen: --fresh: deleting {answers_path}")
            answers_path.unlink()

    spec = MODELS[cfg.model]
    held, n_options = held_questions(meta, pool)
    plan = request_plan(cfg, held, slugs)
    lines, damaged = read_answers(answers_path)
    if damaged and not rescore_only:
        # a job killed while writing: drop the damaged tail, so the members appended next are readable
        log(f"freegen: {answers_path} was damaged (cut or partial line); kept {len(lines)} complete answers")
        write_answers(answers_path, lines, append=False)
    want = expected_vectors(plan, held)
    done = saved_results(lines, plan, max_tokens, want)
    t0 = time.monotonic()
    if rescore_only:
        missing = [k for k, _ in plan if k not in done]
        if missing:
            raise SystemExit(f"freegen --rescore-only: {answers_path} has no {max_tokens}-token answer for "
                             f"{len(missing)} of {len(plan)} requests, e.g. {missing[0]}; run `freegen` (resumes)")
        paths = freegen_vector_paths(cfg, slugs)
        log(f"freegen {run_name}: rescoring {len(plan)} saved answers")
    else:
        tok = load_tokenizer(cfg.model)
        paths = write_freegen_vectors(cfg, slugs)
        reqs = [r for r in build_requests(cfg, held, n_options, paths, tok, max_tokens) if r.key not in done]
        log(f"freegen {run_name}: {len(plan)} requests ({len(held)} questions, traits {slugs}), {max_tokens} new "
            f"tokens, {len(plan) - len(reqs)} saved in {answers_path.name}, {len(reqs)} to generate in chunks of "
            f"{spec.batch_size}")
        if reqs:
            open_engine = open_engine or default_open_engine()
            n = 0
            with open_engine(spec, "steer") as engine:
                for chunk in chunks(reqs, spec.batch_size):
                    by_key = {r.key: r for r in chunk}
                    # only what scoring needs goes to disk: thousands of 15k-token answers' token ids would not fit
                    new = [answer_line(r, by_key[r.key]) for r in engine.generate(chunk) if r.key in by_key]
                    write_answers(answers_path, new)
                    n += len(new)
                    log(f"freegen {run_name}: {len(plan) - len(reqs) + n}/{len(plan)} answers")
        lines, _ = read_answers(answers_path)
        done = saved_results(lines, plan, max_tokens, want)
        missing = [k for k, _ in plan if k not in done]
        if missing:
            raise RuntimeError(f"freegen: the engine returned no answer for {len(missing)} requests, e.g. "
                               f"{missing[0]}")
    unsteered, per_trait = fold(done, held, n_options, slugs)

    # re-read: eval.json may have been rewritten (`eval`) while vLLM was generating
    ev = read_json(root / "eval.json")
    ev["free_gen"] = {"enabled": True, "status": "done", "engine": "vllm-hook (persvecfast.mia steer)",
                      "max_tokens": max_tokens,
                      "sampling": {k: v for k, v in asdict(spec).items() if k in SAMPLING_FIELDS},
                      "presence_penalty_applied": True, "batch_size": spec.batch_size, "n_held": len(held),
                      "instruction": "End your reply with 'ANSWER: <letter>', one of A, B, ...",
                      "answers": str(answers_path.resolve()),
                      "vectors": {s: {k: [str(x) for x in p] if isinstance(p, list) else str(p) for k, p in kv.items()}
                            for s, kv in paths.items()},
                      "random_dirs": RANDOM_DIRS,
                      "unsteered": unsteered}
    for s in slugs:
        ev["traits"][s]["free_gen"] = per_trait[s]
    apply_pass(ev, [s for s in ev["traits"]])
    timing = ev.setdefault("timing", {})
    if rescore_only:
        log(f"freegen {run_name}: rescored; wrote {root / 'eval.json'} (timing.free_gen_s kept)")
    else:
        # a resumed run: this job's seconds only
        timing["free_gen_s"] = round(time.monotonic() - t0, 2)
        log(f"freegen {run_name}: done in {timing['free_gen_s']} s; wrote {root / 'eval.json'}")
    write_json(root / "eval.json", ev)
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
