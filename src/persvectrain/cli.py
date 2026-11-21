"""Command line: `python -m persvectrain {check, reference, train} ...`."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace

import torch
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import log, slugify, write_json

from persvectrain import data as data_mod
from persvectrain.data import full_source_vector, load_pairs, start_vector, supergpqa_prompts, usable_traits
from persvectrain.output import (REPO, build_train_json, prepare_run, run_root, source_dir, write_vectors)
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


@torch.no_grad()
def _generate(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int, **sampling) -> list[str]:
    """Batched HF sampling with left padding; returns the decoded continuations (special tokens dropped)."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        gen = model.generate(**batch, do_sample=True, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **sampling)
        new = gen[:, batch["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
        log(f"reference: {len(outs)}/{len(prompts)} answers")
    return outs


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

    model, tok = load_model(cfg.model)
    vec_dir = prepare_run(cfg)
    root = run_root(cfg.out, cfg.run_name)
    ltype = layer_type(model, cfg.layer)
    log(f"train {cfg.run_name}: {cfg.mode}, traits {kept}, layer {cfg.layer} ({ltype})")

    with open(root / "steps.jsonl", "a", encoding="utf-8") as steps:
        def log_step(entry: dict) -> None:
            steps.write(json.dumps(entry) + "\n")
            steps.flush()

        vectors = train(model, tok, cfg, traits, ref_train, log_step)

    for slug in kept:
        write_vectors(vec_dir, slug, cfg.layer, vectors[slug], src, cfg.run_name, seed=cfg.seed)
    write_json(root / "train.json", build_train_json(cfg, sources, ltype, dropped, splits))
    log(f"train {cfg.run_name}: wrote {root}")
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
    p.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
