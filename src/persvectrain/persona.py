"""`persona`: free answers to the run's held persona questions on vLLM, scored by an OpenRouter judge (spec,
"Persona judge").

Per trait: `unsteered` (neutral system prompt), `elicited` (the trait's eliciting system prompt, no steering), and
`trained`, `source_full`, `trained_random` at c = +1 under the neutral prompt and at c = -1 under the eliciting prompt
(PersVecFast's suppression setting). Thinking off. Answers go to run_root/persona_answers.jsonl.gz (freegen's
format, appended per engine chunk), judge scores (trait expression 0-100 and coherence 0-100, PersVecFast's
`persvecfast.judge`) to run_root/persona_judged.jsonl.gz (appended per judge batch). Both resume: a rerun generates
only the missing answers and judges only the unjudged ones (a None score, e.g. an unparseable judge reply, is
retried). Only the tokenizer is loaded here; vLLM loads the model.

An empty (whitespace-only) answer gets coherence 0 and expression 0 without a judge call.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import time
from pathlib import Path

from persvecfast.engine import Request
from persvecfast.judge import JudgeError, Judges, OpenRouterClient, load_config, score_coherence, \
    score_expression, session
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt, positive_instruction
from persvecfast.util import chunks, load_traits, log, read_json, slugify, write_json

from persvectrain import freegen as fg
from persvectrain.data import load_pairs
from persvectrain.evaluate import FREE_GEN_C, FREE_GEN_KINDS, RANDOM_DIRS, apply_pass, c_key, mean_se
from persvectrain.output import PERSVECFAST_REPO, config_from_train_json, run_root, source_dir
from persvectrain.train import TrainConfig

MAX_TOKENS = 1024
ANSWERS_FILE = "persona_answers.jsonl.gz"
JUDGED_FILE = "persona_judged.jsonl.gz"
UNSTEERED, ELICITED = "unsteered", "elicited"
SEED_LABEL = {UNSTEERED: "neutral", ELICITED: "elicited"}
JUDGE_MODEL = "openai/gpt-oss-120b"
TRAIT_DIR = PERSVECFAST_REPO / "data" / "persona_vectors"


# ---------------------------------------------------------------- inputs


def questions(meta: dict, src: Path) -> list[tuple[str, str]]:
    """[(qid "{slug}:{pair index}", question text)]: the union of every trait's held pair questions of the run,
    ordered by trait (config order) then pair index; the same list for every trait."""
    out = []
    for slug in meta["config"]["traits"]:
        if slug not in meta["splits"]:
            continue
        by_index = {p.index: p for p in load_pairs(src, slug)}
        out += [(f"{slug}:{i}", by_index[i].question) for i in sorted(meta["splits"][slug]["held"])]
    return out


def trait_info(slug: str) -> dict:
    """PersVecFast's trait file (name, description, eliciting instruction) of the slug."""
    names = [t["trait"] for t in load_traits(TRAIT_DIR) if slugify(t["trait"]) == slug]
    if not names:
        raise SystemExit(f"persona: no trait file for {slug} in {TRAIT_DIR}")
    return load_traits(TRAIT_DIR, names)[0]


def conditions() -> list[tuple[str, float]]:
    """(kind, c) per trait in plan order."""
    return [(UNSTEERED, 0.0), (ELICITED, 0.0)] + [(k, c) for k in FREE_GEN_KINDS for c in FREE_GEN_C]


def request_plan(cfg: TrainConfig, qs: list[tuple[str, str]], slugs: list[str]) -> list[tuple[tuple, int]]:
    """(key, seed) of every request: per trait, per condition (unsteered, elicited, then kind x c), per question;
    keys (slug, kind, c, qid). Seeds request_seed(cfg.seed, "persona", c, qi) for steered requests (paired across
    kinds and traits), request_seed(cfg.seed, "persona", "neutral" | "elicited", qi) unsteered / elicited."""
    plan = []
    for slug in slugs:
        for kind, c in conditions():
            label = SEED_LABEL.get(kind, c)
            plan += [((slug, kind, c, qid), request_seed(cfg.seed, "persona", label, qi))
                     for qi, (qid, _) in enumerate(qs)]
    return plan


def system_prompt(kind: str, c: float, trait: dict) -> str:
    """Eliciting prompt for `elicited` and for steered c < 0; neutral otherwise."""
    return positive_instruction(trait, False) if kind == ELICITED or c < 0 else NEUTRAL_SYSTEM_PROMPT


def _pos(qs) -> dict[str, int]:
    return {qid: i for i, (qid, _) in enumerate(qs)}


def vector_names(plan: list[tuple[tuple, int]], qs) -> dict[tuple, str]:
    """The vector file name every steered request must use (trained_random rotates, k = position % 5)."""
    pos = _pos(qs)
    out = {}
    for key, _ in plan:
        slug, kind, _, qid = key
        if kind in (UNSTEERED, ELICITED):
            continue
        out[key] = fg.random_name(slug, pos[qid] % RANDOM_DIRS) if kind == "trained_random" else f"{slug}_{kind}.pt"
    return out


def build_requests(cfg: TrainConfig, qs, paths: dict[str, dict], tok, max_tokens: int,
                   traits: dict[str, dict]) -> list[Request]:
    """The requests of request_plan (traits = paths' keys) with prompts and steering."""
    pos, text = _pos(qs), dict(qs)
    reqs = []
    for key, seed in request_plan(cfg, qs, list(paths)):
        slug, kind, c, qid = key
        prompt = chat_prompt(tok, system_prompt(kind, c, traits[slug]), text[qid], False)
        steer = None if kind in (UNSTEERED, ELICITED) else \
            {"layer": cfg.layer, "coeff": c, "vector_path": str(fg.vector_for(paths, slug, kind, pos[qid]))}
        reqs.append(Request(key, prompt, seed, steer, max_tokens))
    return reqs


# ---------------------------------------------------------------- judged scores (run_root/persona_judged.jsonl.gz)


def read_judged(path: Path) -> list[dict]:
    """The judged lines in file order; a damaged tail or a line without its fields is dropped."""
    if not path.exists():
        return []
    raw, _ = fg._gunzip_tolerant(path.read_bytes())
    out = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and all(k in d for k in ("key", "seed", "score", "coherence")):
            out.append(d)
    return out


def write_judged(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write("".join(json.dumps(ln) + "\n" for ln in lines))


def judged_scores(lines: list[dict], plan, max_tokens: int, vectors: dict[tuple, str]) -> dict[tuple, dict]:
    """{key: line} of the plan's requests judged for their current answer (same seed, max_tokens and vector file;
    the last line wins). A line with a None score or coherence is kept here; `unjudged` retries it."""
    seeds = dict(plan)
    out = {}
    for ln in lines:
        key = tuple(ln["key"])
        if key in seeds and ln["seed"] == seeds[key] and ln.get("max_tokens") == max_tokens \
                and ln.get("vector") == vectors.get(key):
            out[key] = ln
    return out


def unjudged(plan, judged: dict[tuple, dict]) -> list[tuple]:
    return [k for k, _ in plan if k not in judged or judged[k]["score"] is None or judged[k]["coherence"] is None]


def default_make_judges(concurrency: int) -> Judges:
    """OpenRouter judges; the key comes from the env or PersVecFast's .env. RuntimeError when it is missing."""
    from dotenv import load_dotenv

    load_dotenv(PERSVECFAST_REPO / ".env")
    config = load_config(concurrency)
    return Judges(make_client=lambda: OpenRouterClient(config), expression_model=JUDGE_MODEL)


async def _judge_all(judges: Judges, todo: list[dict], write, batch: int) -> int:
    """Judge the items {key, seed, max_tokens, vector, question, text, trait, description} in batches; write(lines)
    after each batch. Raises the first judge error after writing the batch's good results."""
    async def one(it: dict) -> dict:
        line = {"key": list(it["key"]), "seed": it["seed"], "max_tokens": it["max_tokens"], "vector": it["vector"]}
        if not it["text"].strip():
            return {**line, "score": 0, "coherence": 0}
        s, c = await asyncio.gather(
            score_expression(judges, it["question"], it["text"], it["trait"], it["description"]),
            score_coherence(judges, it["question"], it["text"]), return_exceptions=True)
        for r in (s, c):  # both calls have finished: no orphaned call; a JudgeError wins over other errors
            if isinstance(r, JudgeError):
                raise r
        for r in (s, c):
            if isinstance(r, BaseException):
                raise r
        return {**line, "score": s, "coherence": c}

    n = 0
    async with session(judges):
        for part in chunks(todo, batch):
            res = await asyncio.gather(*(one(it) for it in part), return_exceptions=True)
            good = [r for r in res if isinstance(r, dict)]
            write(good)
            n += len(good)
            errs = [r for r in res if isinstance(r, BaseException)]
            if errs:
                raise next((e for e in errs if isinstance(e, JudgeError)), errs[0])
            log(f"persona: judged {n}/{len(todo)}")
    return n


# ---------------------------------------------------------------- folding into eval.json


def block(keys: list[tuple], answers: dict[tuple, object], judged: dict[tuple, dict]) -> dict:
    """One persona_judge block: per-question lists in question order plus means / SEs over the judged ones."""
    score = [judged[k]["score"] for k in keys]
    coh = [judged[k]["coherence"] for k in keys]
    sm, sse = mean_se([float(x) for x in score if x is not None])
    cm, cse = mean_se([float(x) for x in coh if x is not None])
    return {"n": len(keys), "n_scored": sum(x is not None for x in score), "qid": [k[3] for k in keys],
            "score": score, "coherence": coh, "score_mean": sm, "score_se": sse, "coherence_mean": cm,
            "coherence_se": cse, "n_tokens": [int(answers[k].n_tokens) for k in keys]}


def fold(plan, answers, judged, slugs: list[str], qs) -> dict[str, dict]:
    """{slug: {"unsteered": block, "elicited": block, kind: {"c=+1": block, "c=-1": block}}}."""
    def blk(slug, kind, c):
        return block([(slug, kind, c, qid) for qid, _ in qs], answers, judged)

    return {s: {UNSTEERED: blk(s, UNSTEERED, 0.0), ELICITED: blk(s, ELICITED, 0.0),
                **{kind: {c_key(c): blk(s, kind, c) for c in FREE_GEN_C} for kind in FREE_GEN_KINDS}}
            for s in slugs}


# ---------------------------------------------------------------- run


def run_persona(out: Path, run_name: str, max_tokens: int = MAX_TOKENS, open_engine=None, make_judges=None,
                fresh: bool = False, judge_concurrency: int = 32) -> dict | None:
    """Persona answers + judge scores for one finished run; returns its updated eval.json, or None (with a message)
    when the run has no train.json / eval.json, did not finish, was trained with --no-judge, or has no trait.

    make_judges(concurrency) -> Judges (default: OpenRouter, key from the env or PersVecFast's .env) is called
    before anything else, so a missing key stops the job before the engine opens. Answers are appended per engine
    chunk, scores per judge batch; a judge error that retrying cannot fix (no credit, bad key) stops with a message
    and keeps both files, a rerun judges only what is missing. `fresh` deletes both files first. Writes the
    persona files and eval.json's persona_judge parts (and pass rule) only.
    """
    root = run_root(Path(out), run_name)
    for name in ("train.json", "eval.json"):
        if not (root / name).exists():
            print(f"persona: {root / name} missing; nothing to do (run `train` first)", flush=True)
            return None
    meta = read_json(root / "train.json")
    if not meta.get("finished"):
        print(f"persona: run {root} did not finish training (train.json finished false); nothing to do", flush=True)
        return None
    if read_json(root / "eval.json").get("persona_judge", {}).get("status") == "disabled":
        print(f"persona: run {root} was trained with --no-judge (persona_judge status disabled); nothing to do",
              flush=True)
        return None
    cfg = config_from_train_json(meta, out)
    ev_traits = read_json(root / "eval.json")["traits"]
    slugs = [s for s in cfg.traits if s in ev_traits]
    if not slugs:
        print(f"persona: no trait of {root} to answer; nothing to do", flush=True)
        return None
    try:
        judges = (make_judges or default_make_judges)(judge_concurrency)
    except RuntimeError as e:
        raise SystemExit(f"persona: {e}; no answers generated") from None
    traits = {s: trait_info(s) for s in slugs}

    qs = questions(meta, source_dir(cfg))
    plan = request_plan(cfg, qs, slugs)
    vec = vector_names(plan, qs)
    answers_path, judged_path = root / ANSWERS_FILE, root / JUDGED_FILE
    if fresh:
        for p in (answers_path, judged_path):
            if p.exists():
                log(f"persona: --fresh: deleting {p}")
                p.unlink()
    lines, damaged = fg.read_answers(answers_path)
    if damaged:
        log(f"persona: {answers_path} was damaged; kept {len(lines)} complete answers")
        fg.write_answers(answers_path, lines, append=False)
    done = fg.saved_results(lines, plan, max_tokens, vec)
    t0 = time.monotonic()
    spec = MODELS[cfg.model]
    tok = fg.load_tokenizer(cfg.model)
    paths = fg.write_freegen_vectors(cfg, slugs)
    reqs = [r for r in build_requests(cfg, qs, paths, tok, max_tokens, traits) if r.key not in done]
    log(f"persona {run_name}: {len(plan)} requests ({len(qs)} questions, traits {slugs}), {max_tokens} new tokens, "
        f"{len(plan) - len(reqs)} saved, {len(reqs)} to generate")
    if reqs:
        open_engine = open_engine or fg.default_open_engine()
        n = 0
        with open_engine(spec, "steer") as engine:
            for chunk in chunks(reqs, spec.batch_size):
                by_key = {r.key: r for r in chunk}
                new = [fg.answer_line(r, by_key[r.key]) for r in engine.generate(chunk) if r.key in by_key]
                fg.write_answers(answers_path, new)
                n += len(new)
                log(f"persona {run_name}: {len(plan) - len(reqs) + n}/{len(plan)} answers")
    lines, _ = fg.read_answers(answers_path)
    done = fg.saved_results(lines, plan, max_tokens, vec)
    missing = [k for k, _ in plan if k not in done]
    if missing:
        raise RuntimeError(f"persona: the engine returned no answer for {len(missing)} requests, e.g. {missing[0]}")

    judged = judged_scores(read_judged(judged_path), plan, max_tokens, vec)
    todo_keys = unjudged(plan, judged)
    seeds, text = dict(plan), dict(qs)
    items = [{"key": k, "seed": seeds[k], "max_tokens": max_tokens, "vector": vec.get(k), "text": done[k].text,
              "question": text[k[3]], "trait": traits[k[0]]["trait"], "description": traits[k[0]]["description"]}
             for k in todo_keys]
    log(f"persona {run_name}: {len(plan) - len(items)} of {len(plan)} answers judged, {len(items)} to judge")
    if items:
        try:
            asyncio.run(_judge_all(judges, items, lambda ls: write_judged(judged_path, ls),
                                   batch=max(32, 4 * judge_concurrency)))
        except JudgeError as e:
            have = unjudged(plan, judged_scores(read_judged(judged_path), plan, max_tokens, vec))
            raise SystemExit(f"persona: the judge stopped: {e}\n  {len(plan) - len(have)} of {len(plan)} answers "
                             f"judged; answers kept in {answers_path}; fix the cause and rerun `persona` "
                             f"(only the {len(have)} unjudged answers are sent)") from None
    judged = judged_scores(read_judged(judged_path), plan, max_tokens, vec)
    per_trait = fold(plan, done, judged, slugs, qs)

    ev = read_json(root / "eval.json")  # re-read: `eval` may have rewritten it meanwhile
    n_unscored = len(unjudged(plan, judged))
    ev["persona_judge"] = {"status": "done" if n_unscored == 0 else "pending", "judge_model": getattr(judges, "expression_model", JUDGE_MODEL),
                           "max_tokens": max_tokens, "n_questions": len(qs), "answers": str(answers_path.resolve()),
                           "judged": str(judged_path.resolve()), "random_dirs": RANDOM_DIRS,
                           "n_unscored": n_unscored}
    if n_unscored:
        log(f"persona {run_name}: WARNING {n_unscored} of {len(plan)} answers have no judge score (unparseable judge "
            f"replies); persona_judge stays pending; rerun `python -m persvectrain persona --out {out} --run-name "
            f"{run_name}` (only those are judged again)")
    for s in slugs:
        ev["traits"][s]["persona_judge"] = per_trait[s]
    apply_pass(ev, list(ev["traits"]))
    ev.setdefault("timing", {})["persona_s"] = round(time.monotonic() - t0, 2)
    write_json(root / "eval.json", ev)
    log(f"persona {run_name}: done in {ev['timing']['persona_s']} s; wrote {root / 'eval.json'}")
    print_persona_table(ev)
    return ev


def _f(x) -> str:
    return "-" if x is None else f"{x:.1f}"


def print_persona_table(ev: dict) -> None:
    """judge score / coherence (mean+-SE) per trait x condition, then the pass lines."""
    from persvectrain.evaluate import pass_lines

    def cols(b):
        return f"{_f(b['score_mean']):>6}+-{_f(b['score_se']):<5} {_f(b['coherence_mean']):>6}+-{_f(b['coherence_se']):<5}"

    lines = [f"{'trait':<18} {'kind':<16} {'c':>4} {'score':>13} {'coherence':>13}"]
    for s, t in ev["traits"].items():
        pj = t.get("persona_judge")
        if not pj:
            continue
        for kind in (UNSTEERED, ELICITED):
            lines.append(f"{s:<18} {kind:<16} {'0':>4} {cols(pj[kind])}")
        for kind in FREE_GEN_KINDS:
            for c in FREE_GEN_C:
                lines.append(f"{s:<18} {kind:<16} {c_key(c)[2:]:>4} {cols(pj[kind][c_key(c)])}")
    lines += pass_lines(ev)
    print("\n".join(lines), flush=True)
