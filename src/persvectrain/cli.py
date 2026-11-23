"""Command line: `python -m persvectrain {check, reference, train, eval, freegen, kldepth, report} ...`."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import MISSING, fields
from pathlib import Path

import torch
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import log, read_json, slugify, write_json

from persvectrain import data as data_mod
from persvectrain.data import full_source_vector, load_pairs, start_vector, supergpqa_prompts, usable_traits
from persvectrain.evaluate import evaluate, pass_rule
from persvectrain.freegen import MAX_TOKENS as FREEGEN_MAX_TOKENS
from persvectrain.freegen import POOL, run_freegen
from persvectrain.persona import MAX_TOKENS as PERSONA_MAX_TOKENS
from persvectrain.persona import run_persona
from persvectrain.offtarget import run_offtarget
from persvectrain.generate import generate as _generate
from persvectrain.output import (PERSVECFAST_REPO, REPO, build_train_json, config_from_train_json, git_commit,
                                 load_reference, load_trained, prepare_run, run_dir, run_root, source_dir,
                                 write_vectors)
from persvectrain.report import format_table, pick_per_trait, pick_run
from persvectrain.steering import LOGPROB_NORMS, layer_type
from persvectrain.train import RefAnswer, TrainConfig, TraitData, train

REFERENCE_DATA = POOL
REFERENCE_MAX_NEW_TOKENS = 512
REFERENCE_POOL = "supergpqa 200-299 + 340-499"
N_HELD = 100


def load_model(model_key: str):
    """The only model-loading path (tests patch it): bf16 HF model on cuda + its tokenizer."""
    from persvecfast.smoke import load_hf
    from transformers import AutoTokenizer

    repo = MODELS[model_key].hf_repo
    return load_hf(repo), AutoTokenizer.from_pretrained(repo)


def _reference_stem(model: str, tokens: int) -> str:
    """`<model>_pool260`, plus `_t{tokens}` when the answers are longer or shorter than the default 512."""
    return f"{model}_pool260" if tokens == REFERENCE_MAX_NEW_TOKENS else f"{model}_pool260_t{tokens}"


def reference_path(out: Path, model: str, tokens: int = REFERENCE_MAX_NEW_TOKENS) -> Path:
    return Path(out) / "trained" / "reference" / f"{_reference_stem(model, tokens)}.jsonl"


def reference_meta_path(out: Path, model: str, tokens: int = REFERENCE_MAX_NEW_TOKENS) -> Path:
    """Sampling settings and provenance of the reference answers, next to the jsonl."""
    return Path(out) / "trained" / "reference" / f"{_reference_stem(model, tokens)}.meta.json"


# ---------------------------------------------------------------- check


_MODEL_FIELDS = ("model", "extraction_model", "negative_model")


def _model_fields(path: Path) -> str:
    """`; model X, extraction_model Y, ...` from a trait's {slug}.json (the fields present), else ""."""
    if not path.exists():
        return ""
    meta = read_json(path)
    parts = [f"{k} {meta[k]}" for k in _MODEL_FIELDS if meta.get(k) is not None]
    return "; " + ", ".join(parts) if parts else ""


def cmd_check(args) -> int:
    src = Path(args.source) / "persona" / args.model / "bf16"
    slugs = [slugify(t) for t in args.traits]
    kept, dropped = usable_traits(src, slugs, args.min_pairs)
    print(f"source {src}")
    for slug in kept:
        print(f"  {slug}: {len(load_pairs(src, slug))} kept pairs{_model_fields(src / f'{slug}.json')}")
    for slug, reason in dropped.items():
        print(f"  {slug}: dropped ({reason})")
    return 0 if kept else 1


# ---------------------------------------------------------------- reference


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def cmd_reference(args) -> int:
    n_new = args.max_new_tokens
    path = reference_path(args.out, args.model, n_new)
    if path.exists():
        log(f"reference {path} exists; skipping")
        return 0
    spec = MODELS[args.model]
    model, tok = load_model(args.model)
    items = supergpqa_prompts(Path(args.data))
    prompts = [chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, it["question"], False) for it in items]
    log(f"reference: {len(prompts)} prompts, temperature {spec.temperature} top_p {spec.top_p} top_k {spec.top_k}; "
        f"presence_penalty {spec.presence_penalty} is not applied in HF generate")
    seed = request_seed(args.model, "reference")
    torch.manual_seed(seed)
    answers = _generate(model, tok, prompts, args.batch_size, n_new,
                        temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k)
    rows = [{"id": it["id"], "question": it["question"], "answer_text": a, "gold": it["answer"]}
            for it, a in zip(items, answers)]
    meta = {"model": args.model, "hf_repo": spec.hf_repo, "data": str(Path(args.data).resolve()),
            "temperature": spec.temperature, "top_p": spec.top_p, "top_k": spec.top_k,
            "presence_penalty": spec.presence_penalty, "presence_penalty_applied": False, "seed": seed,
            "max_new_tokens": n_new, "batch_size": args.batch_size, "n": len(rows),
            "pool": REFERENCE_POOL,
            "persvectrain_commit": git_commit(REPO), "persvecfast_commit": git_commit(PERSVECFAST_REPO),
            "created": _now()}
    # meta first (atomic): the jsonl is what marks the reference done, so a jsonl never lacks its meta
    write_json(reference_meta_path(args.out, args.model, n_new), meta)
    _write_jsonl_atomic(path, rows)
    log(f"reference: wrote {path}")
    return 0


def split_reference(refs: list[RefAnswer], seed: int,
                    n_held: int | None = None) -> tuple[list[RefAnswer], list[RefAnswer]]:
    """Last n_held of data.split's seeded shuffle are held (260 questions: 160 train, 100 held)."""
    n_held = N_HELD if n_held is None else n_held  # read at call time
    idx = list(range(len(refs)))
    random.Random(seed).shuffle(idx)
    n_train = len(refs) - n_held
    return [refs[i] for i in idx[:n_train]], [refs[i] for i in idx[n_train:]]


# ---------------------------------------------------------------- train


def _config_from_args(args, traits: list[str]) -> TrainConfig:
    kw = {f.name: getattr(args, f.name) for f in fields(TrainConfig)}
    kw["traits"] = traits
    # absolute: train.json's paths stay valid whatever directory `eval` / the games are run from
    kw["source"], kw["out"] = Path(args.source).resolve(), Path(args.out).resolve()
    return TrainConfig(**kw)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def cmd_train(args) -> int:
    cfg = _config_from_args(args, [slugify(t) for t in args.traits])
    src = source_dir(cfg)
    kept, dropped = usable_traits(src, cfg.traits, cfg.min_pairs)
    for slug, reason in dropped.items():
        print(f"dropped trait {slug}: {reason}", flush=True)
    if not kept:
        print("no usable traits left", flush=True)
        return 1
    cfg.traits = kept
    root = run_root(cfg.out, cfg.run_name)
    if root.exists():  # fail before any expensive work; prepare_run checks again
        raise FileExistsError(f"run {root} exists; pick another --run-name")

    ref_file = reference_path(cfg.out, cfg.model, cfg.kl_tokens)
    if not ref_file.exists():
        raise SystemExit(f"reference answers {ref_file} missing: run "
                         f"`python -m persvectrain reference --model {cfg.model} --out {cfg.out} "
                         f"--max-new-tokens {cfg.kl_tokens}` first")
    ref_train, ref_held = split_reference(load_reference(ref_file), cfg.seed)

    traits: dict[str, TraitData] = {}
    sources: dict = {}
    splits: dict = {}
    for slug in kept:
        tr, held = data_mod.split(load_pairs(src, slug), seed=cfg.seed)
        start = start_vector(src, slug, cfg.layer, [p.index for p in tr])
        traits[slug] = TraitData(slug=slug, train=tr, held=held, start=start,
                                 source=full_source_vector(src, slug, cfg.layer))
        sources[slug] = {"responses": str(src / f"{slug}_responses.json"), "raw": str(src / f"{slug}_raw.pt"),
                         "vectors": str(src / f"{slug}.pt"), "json": str(src / f"{slug}.json")}
        splits[slug] = {"train": [p.index for p in tr], "held": [p.index for p in held]}
    sources["reference"] = str(ref_file)
    meta_file = reference_meta_path(cfg.out, cfg.model, cfg.kl_tokens)
    if meta_file.exists():
        sources["reference_meta"] = read_json(meta_file)
    else:
        sources["reference_meta"] = None
        print(f"warning: reference meta {meta_file} missing (sampling settings of the reference unknown); "
              f"continuing", flush=True)
    splits["reference"] = {"train": [r.id for r in ref_train], "held": [r.id for r in ref_held]}

    vec_dir = prepare_run(cfg)
    # written before the model loads and before training (finished false): a killed run keeps its config, seed,
    # commits and splits; layer_type is filled in once the model is loaded
    meta = build_train_json(cfg, sources, None, dropped, splits)
    meta.update(finished=False, started=_now(), ended=None)
    write_json(root / "train.json", meta)
    model, tok = load_model(cfg.model)
    meta["layer_type"] = layer_type(model, cfg.layer)
    write_json(root / "train.json", meta)
    log(f"train {cfg.run_name}: {cfg.mode}, traits {kept}, layer {cfg.layer} ({meta['layer_type']})")

    t0 = time.monotonic()
    with open(root / "steps.jsonl", "a", encoding="utf-8") as steps:
        def log_step(entry: dict) -> None:
            entry = {**entry, "t": round(time.monotonic() - t0, 2)}  # seconds since training start
            steps.write(json.dumps(entry) + "\n")
            steps.flush()

        vectors = train(model, tok, cfg, traits, ref_train, log_step)

    for slug in kept:
        write_vectors(vec_dir, slug, cfg.layer, vectors[slug], src, cfg.run_name)
    # evaluate the vectors as saved (cast to the source dtype), as `eval` on this run would
    evaluate(model, tok, cfg, traits, load_trained(vec_dir, kept, cfg.layer), ref_held,
             free_gen=not args.no_free_gen, persona_judge=not args.no_judge)
    meta.update(finished=True, ended=_now())
    write_json(root / "train.json", meta)
    log(f"train {cfg.run_name}: wrote {root}")
    if not args.no_free_gen:
        log(f"train {cfg.run_name}: free generation pending: run `python -m persvectrain freegen --out {cfg.out} "
            f"--run-name {cfg.run_name}` (vLLM)")
    if not args.no_judge:
        log(f"train {cfg.run_name}: persona judge pending: run `python -m persvectrain persona --out {cfg.out} "
            f"--run-name {cfg.run_name}` (vLLM + OpenRouter key)")
    return 0


# ---------------------------------------------------------------- eval


def cmd_eval(args) -> int:
    """Re-run the teacher-forced evaluation of an existing run from its train.json (config + splits); writes
    eval.json only, keeping an existing eval.json's free-gen block (dropped with --no-free-gen)."""
    root = run_root(Path(args.out), args.run_name)
    if not (root / "train.json").exists():
        raise SystemExit(f"no run at {root} (train.json missing)")
    meta = read_json(root / "train.json")
    cfg = config_from_train_json(meta, args.out)
    src = source_dir(cfg)
    splits = meta["splits"]

    traits: dict[str, TraitData] = {}
    for slug in cfg.traits:
        by_index = {p.index: p for p in load_pairs(src, slug)}
        tr = [by_index[i] for i in splits[slug]["train"]]
        held = [by_index[i] for i in splits[slug]["held"]]
        traits[slug] = TraitData(slug=slug, train=tr, held=held,
                                 start=start_vector(src, slug, cfg.layer, splits[slug]["train"]),
                                 source=full_source_vector(src, slug, cfg.layer))
    ref_file = Path(meta["sources"].get("reference") or reference_path(cfg.out, cfg.model, cfg.kl_tokens))
    refs = {r.id: r for r in load_reference(ref_file)}
    ref_held = [refs[i] for i in splits["reference"]["held"]]
    trained = load_trained(run_dir(cfg.out, cfg.run_name, cfg.model), cfg.traits, cfg.layer)

    model, tok = load_model(cfg.model)
    log(f"eval {cfg.run_name}: traits {cfg.traits}, layer {cfg.layer}")
    previous = read_json(root / "eval.json") if (root / "eval.json").exists() else None
    evaluate(model, tok, cfg, traits, trained, ref_held, free_gen=not args.no_free_gen, previous=previous,
             persona_judge=not args.no_judge)
    return 0


def _add_eval_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-free-gen", action="store_true",
                   help="no free generation for this run (free_gen status disabled: `freegen` skips it and the "
                        "pass rule skips its length and accuracy checks)")
    p.add_argument("--no-judge", action="store_true", dest="no_judge",
                   help="no persona judge for this run (persona_judge status disabled: `persona` skips it and the "
                        "pass rule skips its judge and coherence checks)")


# ---------------------------------------------------------------- persona


def cmd_persona(args) -> int:
    res = run_persona(Path(args.out), args.run_name, max_tokens=args.max_tokens, fresh=args.fresh,
                      judge_concurrency=args.judge_concurrency)
    return 0 if res is not None and res["persona_judge"]["status"] == "done" else 1


def cmd_offtarget(args) -> int:
    cs = tuple(float(c) for c in args.c)
    rc = 0
    for name in args.run_names:
        res = run_offtarget(Path(args.out), name, cs=cs, judge_concurrency=args.judge_concurrency)
        rc |= 0 if res is not None and res["n_unscored"] == 0 else 1
    return rc


# ---------------------------------------------------------------- freegen


def cmd_freegen(args) -> int:
    res = run_freegen(Path(args.out), args.run_name, max_tokens=args.max_tokens, pool=Path(args.pool),
                      fresh=args.fresh, rescore_only=args.rescore_only)
    return 0 if res is not None else 1


# ---------------------------------------------------------------- bench


def cmd_bench(args) -> int:
    from persvectrain.bench import run_bench

    res = run_bench(Path(args.out), args.run_name, questions=Path(args.questions), n=args.n,
                    max_tokens=args.max_tokens, fresh=args.fresh)
    return 0 if res is not None else 1


# ---------------------------------------------------------------- kldepth


def cmd_kldepth(args) -> int:
    from persvectrain.kldepth import run_kldepth

    meta_path = run_root(Path(args.out), args.run_name) / "train.json"
    if not meta_path.exists():
        print(f"kldepth: {meta_path} missing; run `train` first", flush=True)
        return 1
    model_key = read_json(meta_path)["config"]["model"]
    model, tok = load_model(model_key)
    res = run_kldepth(model, tok, Path(args.out), args.run_name, max_tokens=args.max_tokens, n_items=args.n,
                      traits=[slugify(t) for t in args.traits] if args.traits else None, pool=Path(args.pool))
    return 0 if res is not None else 1


# ---------------------------------------------------------------- vecs


def cmd_vecs(args) -> int:
    from persvectrain.vecs import run_vecs

    model = tok = None
    if args.model:
        meta = read_json(run_root(Path(args.out), args.runs[0]) / "train.json")
        model, tok = load_model(meta["config"]["model"])
    run_vecs(Path(args.out), args.runs, model=model, tok=tok, n_items=args.n)
    return 0


# ---------------------------------------------------------------- analysis


def cmd_analysis(args) -> int:
    from persvectrain import analysis as an

    names = args.runs or an.discover_runs(Path(args.out))
    if not names:
        print(f"analysis: no finished run under {Path(args.out) / 'trained'}")
        return 1
    runs = [an.load_run(Path(args.out), r) for r in names]
    rows = an.comparisons(runs)
    rand = [x for r in runs for x in an.random_direction_rows(r)]
    rot = None if args.no_vectors else an.rotation(Path(args.out), names)
    print(an.format_report(rows, rot, args.z, rand))
    return 0


# ---------------------------------------------------------------- report


def cmd_report(args) -> int:
    evals: dict[str, dict] = {}
    for run in args.runs:
        path = run_root(Path(args.out), run) / "eval.json"
        if not path.exists():
            print(f"{run}: no eval.json")
            continue
        evals[run] = read_json(path)
        # the pass flags are recomputed with the current rule from the stored per-item results (an eval.json written
        # under an older rule, e.g. coherence against the source, is judged by today's rule; old formats keep theirs)
        for t in evals[run]["traits"].values():
            try:
                t["pass"] = pass_rule(t)
            except (KeyError, TypeError, ValueError):
                pass
    if evals:
        print(format_table(evals))
    from persvectrain.bench import OUT_FILE as BENCH_FILE, format_bench
    for run in evals:
        bpath = run_root(Path(args.out), run) / BENCH_FILE
        if bpath.exists():
            print(format_bench(read_json(bpath)))
    best = pick_run(evals)
    print(f"picked: {best}" if best else "no run passes")
    if args.per_trait or args.write_profile:
        picks = pick_per_trait(evals)
        for trait, run in picks.items():
            print(f"{trait} -> {run}" if run else f"{trait} -> none passes")
        if args.write_profile:
            from persvecfast.extract import vector_file
            profile = {}
            for trait, run in picks.items():
                if run is None:
                    profile[trait] = None
                    continue
                ev = evals[run]
                vec = vector_file(run_dir(Path(args.out), run, ev["model"]), trait, ev["layer"])
                profile[trait] = {"run": run, "vector": str(vec)}
            write_json(Path(args.write_profile), profile)
            print(f"wrote {args.write_profile}")
    return 0


# ---------------------------------------------------------------- parser

_REQUIRED_TYPES = {"model": str, "layer": int, "mode": str, "source": str, "out": str, "run_name": str}
_CHOICES = {"mode": ("solo", "joint"), "init": ("dom", "zero"), "logprob_norm": LOGPROB_NORMS, "fix_norm": (0, 1)}


def _add_train_flags(p: argparse.ArgumentParser) -> None:
    """One flag per TrainConfig field (`--lambda-kl` for lambda_kl, ...)."""
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "traits":
            p.add_argument(flag, nargs="+", required=True, help="display names, e.g. Assertive")
            continue
        kw: dict = {"dest": f.name}
        if f.name in _CHOICES:
            kw["choices"] = _CHOICES[f.name]
        if f.default is MISSING:
            kw.update(type=_REQUIRED_TYPES[f.name], required=f.name != "mode")
            if f.name == "mode":
                kw["default"] = "solo"
        else:
            kw.update(type=type(f.default), default=f.default)
        p.add_argument(flag, **kw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="persvectrain")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="kept pairs per trait and the dropped traits")
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True, help="PersVecFast out root")
    p.add_argument("--traits", nargs="+", required=True)
    p.add_argument("--min-pairs", type=int, default=20)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("reference", help="unsteered SuperGPQA 200-299 answers for the KL term")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="out root; writes <out>/trained/reference/<model>.jsonl")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=REFERENCE_MAX_NEW_TOKENS, dest="max_new_tokens",
                   help="new tokens per answer; other than 512 writes <model>_pool260_t{N}.jsonl")
    p.add_argument("--data", default=str(REFERENCE_DATA))
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("train", help="train steering vectors")
    _add_train_flags(p)
    _add_eval_flags(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="re-run the evaluation of an existing run (overwrites its eval.json only)")
    p.add_argument("--out", required=True, help="out root holding trained/<run_name>")
    p.add_argument("--run-name", required=True, dest="run_name")
    _add_eval_flags(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("vecs", help="what training changed in the vectors (trained - start), shared across traits?")
    p.add_argument("--out", required=True, help="out root holding trained/<run_name>")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--model", action="store_true",
                   help="also load the model (GPU): logit lens and layer residual geometry")
    p.add_argument("--n", type=int, default=40, help="KL reference answers for the residual geometry (with --model)")
    p.set_defaults(func=cmd_vecs)

    from persvectrain.bench import BENCH
    p = sub.add_parser("bench", help="larger SuperGPQA accuracy check of a finished run on 500 new questions "
                                     "(writes bench.json; not part of the pass rule)")
    p.add_argument("--out", required=True, help="out root holding trained/<run_name>")
    p.add_argument("--run-name", required=True, dest="run_name")
    p.add_argument("--questions", default=str(BENCH), help="bench question file (scripts/make_bench.py)")
    p.add_argument("--n", type=int, default=None, help="only the first n questions (default all)")
    p.add_argument("--max-tokens", type=int, default=None, help=f"max new tokens (default {FREEGEN_MAX_TOKENS})")
    p.add_argument("--fresh", action="store_true", help="delete the run's bench_answers.jsonl.gz first")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("freegen", help="free-generation check of a finished run on vLLM (fills eval.json's "
                                       "free_gen block and recomputes the pass rule)")
    p.add_argument("--out", required=True, help="out root holding trained/<run_name>")
    p.add_argument("--run-name", required=True, dest="run_name")
    p.add_argument("--max-tokens", type=int, default=None,
                   help=f"max new tokens per answer (default {FREEGEN_MAX_TOKENS}; with --rescore-only the cap "
                        "eval.json records)")
    p.add_argument("--pool", default=str(POOL), help="the SuperGPQA pool file (option counts per question)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fresh", action="store_true",
                   help="delete the run's freegen_answers.jsonl.gz first (default: resume, reuse saved answers)")
    g.add_argument("--rescore-only", action="store_true", dest="rescore_only",
                   help="no engine: re-parse the saved answers and rewrite eval.json's free-gen parts")
    p.set_defaults(func=cmd_freegen)

    p = sub.add_parser("persona", help="persona answers of a finished run on vLLM, scored by an OpenRouter judge "
                                       "(key from OPENROUTER_API_KEY or PersVecFast's .env); fills eval.json's "
                                       "persona_judge block and recomputes the pass rule")
    p.add_argument("--out", required=True, help="out root holding trained/<run_name>")
    p.add_argument("--run-name", required=True, dest="run_name")
    p.add_argument("--max-tokens", type=int, default=PERSONA_MAX_TOKENS, help="max new tokens per answer")
    p.add_argument("--judge-concurrency", type=int, default=32, dest="judge_concurrency")
    p.add_argument("--fresh", action="store_true",
                   help="delete the run's persona_answers / persona_judged files first (default: resume)")
    p.set_defaults(func=cmd_persona)

    p = sub.add_parser("offtarget", help="judge a run's saved persona answers of trait s for every OTHER trait of "
                                         "the run (no engine; OpenRouter judge as `persona`); writes "
                                         "offtarget.json, eval.json untouched")
    p.add_argument("--out", required=True, help="out root holding trained/<run_name>")
    p.add_argument("--run-names", required=True, nargs="+", dest="run_names")
    p.add_argument("--c", nargs="+", default=["1"],
                   help="steering signs to judge (default 1: at -1 the answers carry s's eliciting prompt)")
    p.add_argument("--judge-concurrency", type=int, default=32, dest="judge_concurrency")
    p.set_defaults(func=cmd_offtarget)

    p = sub.add_parser("kldepth", help="diagnostic: KL by answer position (0-512, 512-2048, ...) on the run's "
                                       "saved unsteered free answers; writes kl_depth.json only")
    p.add_argument("--out", required=True, help="out root holding trained/<run_name>")
    p.add_argument("--run-name", required=True, dest="run_name")
    p.add_argument("--max-tokens", type=int, default=8192, help="answer tokens read per answer (default 8192)")
    p.add_argument("--n", type=int, default=None, help="first n held answers only (default all)")
    p.add_argument("--traits", nargs="+", default=None, help="display names (default all of the run's)")
    p.add_argument("--pool", default=str(POOL), help="the SuperGPQA pool file")
    p.set_defaults(func=cmd_kldepth)

    p = sub.add_parser("analysis", help="every paired trained-vs-control comparison (bench, free answers, judge, "
                                        "teacher-forced) summed over traits, the cells where trained is worse than "
                                        "source, and the rotation; reads only")
    p.add_argument("--out", required=True, help="out root holding trained/<run>")
    p.add_argument("--runs", nargs="+", default=None, help="default: every finished run under <out>/trained")
    p.add_argument("--z", type=float, default=2.0, help="|z| at which a trait cell counts as better / worse")
    p.add_argument("--no-vectors", action="store_true", dest="no_vectors",
                   help="leave out the rotation part (it reads the source *_raw.pt files)")
    p.set_defaults(func=cmd_analysis)

    p = sub.add_parser("report", help="pass/fail table over runs and the best run")
    p.add_argument("--out", required=True, help="out root holding trained/<run>/eval.json")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--per-trait", action="store_true", dest="per_trait",
                   help="also the best run per trait (passing, largest mean trained margin at c=+-1)")
    p.add_argument("--write-profile", default=None, dest="write_profile",
                   help="write {trait: {run, vector}} (null where no run passes) to this json file")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
