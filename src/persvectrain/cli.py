"""Command line: `python -m persvectrain {check, reference, train, eval, report} ...`."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import torch
from persvecfast.extract import vector_file
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import log, read_json, slugify, write_json

from persvectrain import data as data_mod
from persvectrain.data import full_source_vector, load_pairs, start_vector, supergpqa_prompts, usable_traits
from persvectrain.evaluate import evaluate
from persvectrain.generate import generate as _generate
from persvectrain.output import (REPO, build_train_json, prepare_run, run_dir, run_root, source_dir,
                                 write_vectors)
from persvectrain.report import format_table, pick_run
from persvectrain.steering import layer_type
from persvectrain.train import RefAnswer, TrainConfig, TraitData, train

REFERENCE_DATA = REPO / "data" / "supergpqa_200_299.jsonl"
REFERENCE_MAX_NEW_TOKENS = 512


def load_model(model_key: str):
    """The only model-loading path (tests patch it): bf16 HF model on cuda + its tokenizer."""
    from persvecfast.smoke import load_hf
    from transformers import AutoTokenizer

    repo = MODELS[model_key].hf_repo
    return load_hf(repo), AutoTokenizer.from_pretrained(repo)


def reference_path(out: Path, model: str) -> Path:
    return Path(out) / "trained" / "reference" / f"{model}.jsonl"


# ---------------------------------------------------------------- check


def cmd_check(args) -> int:
    src = Path(args.source) / "persona" / args.model / "bf16"
    slugs = [slugify(t) for t in args.traits]
    kept, dropped = usable_traits(src, slugs, args.min_pairs)
    print(f"source {src}")
    for slug in kept:
        print(f"  {slug}: {len(load_pairs(src, slug))} kept pairs")
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
    path = reference_path(args.out, args.model)
    if path.exists():
        log(f"reference {path} exists; skipping")
        return 0
    spec = MODELS[args.model]
    model, tok = load_model(args.model)
    items = supergpqa_prompts(Path(args.data))
    prompts = [chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, it["question"], False) for it in items]
    log(f"reference: {len(prompts)} prompts, temperature {spec.temperature} top_p {spec.top_p} top_k {spec.top_k}; "
        f"presence_penalty {spec.presence_penalty} is not applied in HF generate")
    torch.manual_seed(request_seed(args.model, "reference"))
    answers = _generate(model, tok, prompts, args.batch_size, REFERENCE_MAX_NEW_TOKENS,
                        temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k)
    rows = [{"id": it["id"], "question": it["question"], "answer_text": a, "gold": it["answer"]}
            for it, a in zip(items, answers)]
    _write_jsonl_atomic(path, rows)
    log(f"reference: wrote {path}")
    return 0


def load_reference(path: Path) -> list[RefAnswer]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [RefAnswer(id=r["id"], question=r["question"], answer_text=r["answer_text"], gold=r["gold"])
            for r in rows]


def split_reference(refs: list[RefAnswer], seed: int) -> tuple[list[RefAnswer], list[RefAnswer]]:
    """80/20 by file position, with data.split's seeded algorithm."""
    wrapped = [SimpleNamespace(index=i, ref=r) for i, r in enumerate(refs)]
    tr, held = data_mod.split(wrapped, seed=seed)
    return [w.ref for w in tr], [w.ref for w in held]


# ---------------------------------------------------------------- train


def _config_from_args(args, traits: list[str]) -> TrainConfig:
    kw = {f.name: getattr(args, f.name) for f in fields(TrainConfig)}
    kw["traits"] = traits
    kw["source"], kw["out"] = Path(args.source), Path(args.out)
    return TrainConfig(**kw)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def cmd_train(args) -> int:
    cfg = _config_from_args(args, [slugify(t) for t in args.traits])
    src = source_dir(cfg)
    kept, dropped = usable_traits(src, cfg.traits)
    for slug, reason in dropped.items():
        print(f"dropped trait {slug}: {reason}", flush=True)
    if not kept:
        print("no usable traits left", flush=True)
        return 1
    cfg.traits = kept
    root = run_root(cfg.out, cfg.run_name)
    if root.exists():  # fail before any expensive work; prepare_run checks again
        raise FileExistsError(f"run {root} exists; pick another --run-name")

    ref_file = reference_path(cfg.out, cfg.model)
    if not ref_file.exists():
        raise SystemExit(f"reference answers {ref_file} missing: run "
                         f"`python -m persvectrain reference --model {cfg.model} --out {cfg.out}` first")
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
        write_vectors(vec_dir, slug, cfg.layer, vectors[slug], src, cfg.run_name, seed=cfg.seed)
    # evaluate the vectors as saved (cast to the source dtype), as `eval` on this run would
    evaluate(model, tok, cfg, traits, load_trained(vec_dir, kept, cfg.layer), ref_held,
             free_gen=not args.no_free_gen, free_gen_tokens=args.free_gen_tokens)
    meta.update(finished=True, ended=_now())
    write_json(root / "train.json", meta)
    log(f"train {cfg.run_name}: wrote {root}")
    return 0


# ---------------------------------------------------------------- eval


def load_trained(vec_dir: Path, slugs: list[str], layer: int) -> dict[str, torch.Tensor]:
    """The run's trained vectors from vllm_hook/{slug}_L{layer}.pt, as fp32."""
    out = {}
    for slug in slugs:
        path = vector_file(vec_dir, slug, layer)
        if not path.exists():
            raise SystemExit(f"trained vector {path} missing: training did not finish writing {slug}")
        out[slug] = torch.load(path)["dir"].float()
    return out


def cmd_eval(args) -> int:
    """Re-run the evaluation of an existing run from its train.json (config + splits); writes eval.json only."""
    root = run_root(Path(args.out), args.run_name)
    if not (root / "train.json").exists():
        raise SystemExit(f"no run at {root} (train.json missing)")
    meta = read_json(root / "train.json")
    kw = dict(meta["config"])
    kw["source"], kw["out"] = Path(kw["source"]), Path(args.out)
    cfg = TrainConfig(**kw)
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
    ref_file = Path(meta["sources"].get("reference") or reference_path(cfg.out, cfg.model))
    refs = {r.id: r for r in load_reference(ref_file)}
    ref_held = [refs[i] for i in splits["reference"]["held"]]
    trained = load_trained(run_dir(cfg.out, cfg.run_name, cfg.model), cfg.traits, cfg.layer)

    model, tok = load_model(cfg.model)
    log(f"eval {cfg.run_name}: traits {cfg.traits}, layer {cfg.layer}")
    evaluate(model, tok, cfg, traits, trained, ref_held, free_gen=not args.no_free_gen,
             free_gen_tokens=args.free_gen_tokens)
    return 0


def _add_eval_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-free-gen", action="store_true",
                   help="skip free generation (the pass rule then skips its length checks)")
    p.add_argument("--free-gen-tokens", type=int, default=4096, help="max new tokens per free answer")


# ---------------------------------------------------------------- report


def cmd_report(args) -> int:
    evals: dict[str, dict] = {}
    for run in args.runs:
        path = run_root(Path(args.out), run) / "eval.json"
        if not path.exists():
            print(f"{run}: no eval.json")
            continue
        evals[run] = read_json(path)
    if evals:
        print(format_table(evals))
    best = pick_run(evals)
    print(f"picked: {best}" if best else "no run passes")
    return 0


# ---------------------------------------------------------------- parser

_REQUIRED_TYPES = {"model": str, "layer": int, "mode": str, "source": str, "out": str, "run_name": str}
_CHOICES = {"mode": ("solo", "joint"), "init": ("dom", "zero")}


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

    p = sub.add_parser("report", help="pass/fail table over runs and the best run")
    p.add_argument("--out", required=True, help="out root holding trained/<run>/eval.json")
    p.add_argument("--runs", nargs="+", required=True)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
