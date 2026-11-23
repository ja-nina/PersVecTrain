"""`offtarget`: does steering one trait move the others? The saved persona answers (persona_answers.jsonl.gz) of
trait s are judged for every OTHER trait t of the run, with the same judge and expression prompt as `persona`.

No engine, no new answers: only judge calls. Judged per trait s: `unsteered` and every steered kind (trained,
source_full, source_at_trained_norm, trained_random) at the chosen c (default +1 only: at c = -1 the answers were
written under s's eliciting system prompt, so a shift in t there is mostly the prompt's). Scores are appended per
judge batch to run_root/offtarget_judged.jsonl.gz (resumes; a None score is retried) and folded into
run_root/offtarget.json:
  traits[s][t][label] = {"qid": [...], "score": [...], "score_mean": ...}, label "unsteered" or "<kind>@c=+1".
eval.json is not touched.
"""
from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

from persvecfast.judge import JudgeError, Judges, score_expression, session
from persvecfast.util import chunks, log, read_json, write_json

from persvectrain import freegen as fg
from persvectrain import persona as pm
from persvectrain.evaluate import c_key, mean_se
from persvectrain.output import config_from_train_json, run_root, source_dir

JUDGED_FILE = "offtarget_judged.jsonl.gz"
OUT_FILE = "offtarget.json"


def label(kind: str, c: float) -> str:
    return pm.UNSTEERED if kind == pm.UNSTEERED else f"{kind}@{c_key(c)}"


def wanted(plan: list[tuple[tuple, int]], cs: tuple[float, ...]) -> list[tuple]:
    """The persona keys to judge: unsteered and the steered kinds at c in cs (never elicited)."""
    return [k for k, _ in plan
            if k[1] == pm.UNSTEERED or (k[1] in pm.KINDS and k[2] in cs)]


def read_judged(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw, _ = fg._gunzip_tolerant(path.read_bytes())
    out = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and all(k in d for k in ("key", "other", "seed", "score")):
            out.append(d)
    return out


def write_judged(path: Path, lines: list[dict]) -> None:
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write("".join(json.dumps(ln) + "\n" for ln in lines))


def judged_scores(lines, seeds: dict, max_tokens: int, vectors: dict) -> dict[tuple, int | None]:
    """{(key, other): score} for lines that match the current answer (seed, max_tokens, vector); last wins."""
    out = {}
    for ln in lines:
        key = tuple(ln["key"])
        if key in seeds and ln["seed"] == seeds[key] and ln.get("max_tokens") == max_tokens \
                and ln.get("vector") == vectors.get(key):
            out[(key, ln["other"])] = ln["score"]
    return out


async def _judge_all(judges: Judges, todo: list[dict], write, batch: int) -> int:
    async def one(it: dict) -> dict:
        line = {"key": list(it["key"]), "other": it["other"], "seed": it["seed"], "max_tokens": it["max_tokens"],
                "vector": it["vector"]}
        if not it["text"].strip():
            return {**line, "score": 0}
        return {**line, "score": await score_expression(judges, it["question"], it["text"], it["trait"],
                                                        it["description"])}

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
            log(f"offtarget: judged {n}/{len(todo)}")
    return n


def run_offtarget(out: Path, run_name: str, cs: tuple[float, ...] = (1.0,), make_judges=None,
                  judge_concurrency: int = 32) -> dict | None:
    """Judge the run's saved persona answers for every other trait; returns offtarget.json's content, or None
    (with a message) when the run has no finished persona answers."""
    root = run_root(Path(out), run_name)
    ev_path, ans_path = root / "eval.json", root / pm.ANSWERS_FILE
    if not (root / "train.json").exists() or not ev_path.exists() or not ans_path.exists():
        print(f"offtarget: {root} has no train.json / eval.json / {pm.ANSWERS_FILE}; run `persona` first", flush=True)
        return None
    pj = read_json(ev_path).get("persona_judge") or {}
    if pj.get("status") != "done":
        print(f"offtarget: {run_name}: persona_judge is not done (status {pj.get('status')}); run `persona` first",
              flush=True)
        return None
    meta = read_json(root / "train.json")
    cfg = config_from_train_json(meta, out)
    ev_traits = read_json(ev_path)["traits"]
    slugs = [s for s in cfg.traits if s in ev_traits]
    if len(slugs) < 2:
        print(f"offtarget: {run_name} has fewer than 2 traits; nothing to do", flush=True)
        return None
    max_tokens = pj["max_tokens"]
    try:
        judges = (make_judges or pm.default_make_judges)(judge_concurrency)
    except RuntimeError as e:
        raise SystemExit(f"offtarget: {e}") from None
    traits = {s: pm.trait_info(s) for s in slugs}
    qs = pm.questions(meta, source_dir(cfg))
    plan = pm.request_plan(cfg, qs, slugs)
    seeds, vec, text = dict(plan), pm.vector_names(plan, qs), dict(qs)
    lines, _ = fg.read_answers(ans_path)
    done = fg.saved_results(lines, plan, max_tokens, vec)
    keys = wanted(plan, cs)
    missing = [k for k in keys if k not in done]
    if missing:
        raise SystemExit(f"offtarget: {len(missing)} persona answers missing (e.g. {missing[0]}); rerun `persona`")

    jpath = root / JUDGED_FILE
    judged = judged_scores(read_judged(jpath), seeds, max_tokens, vec)
    todo = [{"key": k, "other": t, "seed": seeds[k], "max_tokens": max_tokens, "vector": vec.get(k),
             "text": done[k].text, "question": text[k[3]], "trait": traits[t]["trait"],
             "description": traits[t]["description"]}
            for k in keys for t in slugs if t != k[0] and judged.get((k, t)) is None]
    n_all = len(keys) * (len(slugs) - 1)
    log(f"offtarget {run_name}: {n_all - len(todo)} of {n_all} judged, {len(todo)} to judge "
        f"({len(keys)} answers x {len(slugs) - 1} other traits)")
    if todo:
        try:
            asyncio.run(_judge_all(judges, todo, lambda ls: write_judged(jpath, ls),
                                   batch=max(32, 4 * judge_concurrency)))
        except JudgeError as e:
            raise SystemExit(f"offtarget: the judge stopped: {e}; rerun to judge only the rest") from None
    judged = judged_scores(read_judged(jpath), seeds, max_tokens, vec)

    res = {"judge_model": getattr(judges, "expression_model", pm.JUDGE_MODEL), "max_tokens": max_tokens,
           "c": [c_key(c) for c in cs], "n_questions": len(qs), "traits": {}}
    n_none = 0
    for s in slugs:
        res["traits"][s] = {}
        groups: dict[str, list[tuple]] = {}
        for k in keys:
            if k[0] == s:
                groups.setdefault(label(k[1], k[2]), []).append(k)
        for t in slugs:
            if t == s:
                continue
            res["traits"][s][t] = {}
            for lab, ks in groups.items():
                sc = [judged.get((k, t)) for k in ks]
                n_none += sum(x is None for x in sc)
                m, se = mean_se([float(x) for x in sc if x is not None])
                res["traits"][s][t][lab] = {"qid": [k[3] for k in ks], "score": sc, "score_mean": m, "score_se": se}
    res["n_unscored"] = n_none
    write_json(root / OUT_FILE, res)
    log(f"offtarget {run_name}: wrote {root / OUT_FILE}" + (f" ({n_none} unscored; rerun)" if n_none else ""))
    return res
