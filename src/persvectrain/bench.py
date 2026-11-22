"""`bench`: a larger SuperGPQA accuracy check of a finished run on questions it never saw.

The 100 held-out pool questions give an accuracy SE of about 0.04, as large as the effects of interest. `bench`
answers `data/supergpqa_bench500.jsonl` (scripts/make_bench.py: the next 500 'hard' questions of the same seed-0
shuffle as MA_Environments' supergpqa.jsonl, so none of the games' / careful-reasoning / KL-pool questions) exactly
like `freegen`: vLLM-Hook steer engine, once unsteered, and per trait `trained`, `source_full` and `trained_random`
(5 random directions, question i uses k = i % 5) at c = +1 and -1, neutral system prompt, thinking off as freegen,
the games' max_tokens and the model card's sampling; one seed per (c, question) for every steered kind (paired).

Answers go to run_root/bench_answers.jsonl.gz (resumable, as freegen's file); results to run_root/bench.json with
per-question lists and paired accuracy differences (trained - unsteered, source_full - unsteered, trained -
source_full, trained_random - unsteered). eval.json, train.json and the run's vectors are never touched; bench is
not part of the pass rule.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from persvecfast.models import MODELS
from persvecfast.util import chunks, log, read_json, write_json

from persvectrain import freegen as fg
from persvectrain.data import supergpqa_prompts
from persvectrain.evaluate import FREE_GEN_C, RANDOM_DIRS, c_key, mean_se
from persvectrain.output import REPO, config_from_train_json, run_root
from persvectrain.train import RefAnswer

BENCH = REPO / "data" / "supergpqa_bench500.jsonl"
ANSWERS_FILE = "bench_answers.jsonl.gz"
OUT_FILE = "bench.json"
# (name, a, b): paired accuracy difference a - b per (trait, c)
DIFFS = (("trained-unsteered", "trained", "unsteered"), ("source-unsteered", "source_full", "unsteered"),
         ("trained-source", "trained", "source_full"), ("random-unsteered", "trained_random", "unsteered"))


def bench_questions(path: Path, n: int | None = None) -> tuple[list[RefAnswer], dict[str, int]]:
    """The bench questions (first n, file order) as RefAnswer (question = the pool's prompt text) and option counts."""
    items = supergpqa_prompts(Path(path))[:n]
    n_options = fg.pool_options(Path(path))
    return ([RefAnswer(id=it["id"], question=it["question"], answer_text="", gold=it["answer"]) for it in items],
            {it["id"]: n_options[it["id"]] for it in items})


def _correct(block: dict) -> list[float]:
    return [float(x) for x in block["correct"]]


def diffs(unsteered: dict, trait_fg: dict) -> dict:
    """{c: {name: {"mean", "se"}}} of the paired accuracy differences (DIFFS) at c = +1 and -1."""
    out = {}
    for c in FREE_GEN_C:
        k = c_key(c)
        blk = {"unsteered": unsteered, **{kind: trait_fg[kind][k] for kind in ("trained", "source_full",
                                                                            "trained_random")}}
        if any(b["qid"] != unsteered["qid"] for b in blk.values()):
            raise ValueError(f"bench answers at {k} are not paired: question ids differ between kinds")
        out[k] = {}
        for name, a, b in DIFFS:
            m, se = mean_se([x - y for x, y in zip(_correct(blk[a]), _correct(blk[b]))])
            out[k][name] = {"mean": m, "se": se}
    return out


def run_bench(out: Path, run_name: str, questions: Path = BENCH, n: int | None = None,
              max_tokens: int | None = None, open_engine=None, fresh: bool = False) -> dict | None:
    """Bench one finished run; returns bench.json's content, or None (with a message) when there is nothing to do."""
    root = run_root(Path(out), run_name)
    for name in ("train.json", "eval.json"):
        if not (root / name).exists():
            print(f"bench: {root / name} missing; nothing to do (run `train` first)", flush=True)
            return None
    meta = read_json(root / "train.json")
    if not meta.get("finished"):
        print(f"bench: run {root} did not finish training (train.json finished false); nothing to do", flush=True)
        return None
    cfg = config_from_train_json(meta, out)
    ev_traits = read_json(root / "eval.json")["traits"]
    slugs = [s for s in cfg.traits if s in ev_traits]
    if not slugs:
        print(f"bench: no trait of {root} to answer; nothing to do", flush=True)
        return None
    held, n_options = bench_questions(questions, n)
    ref_ids = set(sum((list(v) for v in (meta.get("splits", {}).get("reference") or {}).values()), []))
    overlap = [q.id for q in held if q.id in ref_ids]
    if overlap:
        raise SystemExit(f"bench: {len(overlap)} bench questions are in the run's KL reference split, e.g. "
                         f"{overlap[0]}; the bench set must be new questions")
    max_tokens = fg.MAX_TOKENS if max_tokens is None else max_tokens
    answers_path = root / ANSWERS_FILE
    if fresh and answers_path.exists():
        log(f"bench: --fresh: deleting {answers_path}")
        answers_path.unlink()

    spec = MODELS[cfg.model]
    plan = fg.request_plan(cfg, held, slugs)
    lines, damaged = fg.read_answers(answers_path)
    if damaged:
        log(f"bench: {answers_path} was damaged (cut or partial line); kept {len(lines)} complete answers")
        fg.write_answers(answers_path, lines, append=False)
    want = fg.expected_vectors(plan, held)
    done = fg.saved_results(lines, plan, max_tokens, want)
    t0 = time.monotonic()
    paths = fg.freegen_vector_paths(cfg, slugs)
    if not all(p.exists() for kv in paths.values() for v in kv.values() for p in (v if isinstance(v, list) else [v])):
        paths = fg.write_freegen_vectors(cfg, slugs)  # the same files freegen writes (deterministic)
    tok = fg.load_tokenizer(cfg.model)
    reqs = [r for r in fg.build_requests(cfg, held, n_options, paths, tok, max_tokens) if r.key not in done]
    log(f"bench {run_name}: {len(plan)} requests ({len(held)} questions, traits {slugs}), {max_tokens} new tokens, "
        f"{len(plan) - len(reqs)} saved in {answers_path.name}, {len(reqs)} to generate in chunks of "
        f"{spec.batch_size}")
    if reqs:
        open_engine = open_engine or fg.default_open_engine()
        k = 0
        with open_engine(spec, "steer") as engine:
            for chunk in chunks(reqs, spec.batch_size):
                by_key = {r.key: r for r in chunk}
                new = [fg.answer_line(r, by_key[r.key]) for r in engine.generate(chunk) if r.key in by_key]
                fg.write_answers(answers_path, new)
                k += len(new)
                log(f"bench {run_name}: {len(plan) - len(reqs) + k}/{len(plan)} answers")
    lines, _ = fg.read_answers(answers_path)
    done = fg.saved_results(lines, plan, max_tokens, want)
    missing = [key for key, _ in plan if key not in done]
    if missing:
        raise RuntimeError(f"bench: the engine returned no answer for {len(missing)} requests, e.g. {missing[0]}")
    unsteered, per_trait = fg.fold(done, held, n_options, slugs)
    res = {"run_name": run_name, "questions": str(Path(questions).resolve()), "n": len(held),
           "engine": "vllm-hook (persvecfast.mia steer)", "max_tokens": max_tokens,
           "sampling": {k: v for k, v in asdict(spec).items() if k in fg.SAMPLING_FIELDS},
           "random_dirs": RANDOM_DIRS, "answers": str(answers_path.resolve()), "unsteered": unsteered,
           "traits": {s: {**{kind: v for kind, v in per_trait[s].items() if kind != "unsteered"},
                          "diffs": diffs(unsteered, per_trait[s])} for s in slugs},
           "timing": {"bench_s": round(time.monotonic() - t0, 2)}}
    write_json(root / OUT_FILE, res)
    log(f"bench {run_name}: wrote {root / OUT_FILE}")
    print(format_bench(res), flush=True)
    return res


def _pm(d: dict) -> str:
    return "-" if d["mean"] is None else f"{d['mean']:+.3f}+-{(d['se'] or 0):.3f}"


def format_bench(res: dict) -> str:
    """Accuracy per trait x c (unsteered, trained, source, random) and the paired differences with their SE."""
    u = res["unsteered"]["accuracy"]
    lines = [f"bench {res['run_name']}: {res['n']} SuperGPQA questions, unsteered accuracy {u:.3f}",
             f"{'trait':<15} {'c':>3} {'trained':>8} {'source':>7} {'random':>7}   {'trained-unst':>14} "
             f"{'source-unst':>14} {'trained-source':>14} {'random-unst':>14}"]
    for s, t in res["traits"].items():
        for c in FREE_GEN_C:
            k = c_key(c)
            d = t["diffs"][k]
            lines.append(f"{s:<15} {k[2:]:>3} {t['trained'][k]['accuracy']:>8.3f} {t['source_full'][k]['accuracy']:>7.3f} "
                         f"{t['trained_random'][k]['accuracy']:>7.3f}   {_pm(d['trained-unsteered']):>14} "
                         f"{_pm(d['source-unsteered']):>14} {_pm(d['trained-source']):>14} "
                         f"{_pm(d['random-unsteered']):>14}")
    return "\n".join(lines)
