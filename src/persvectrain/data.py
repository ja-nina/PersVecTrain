from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Pair:
    index: int
    question: str
    pos: str
    neg: str


def load_pairs(src: Path, slug: str) -> list[Pair]:
    data = json.loads((Path(src) / f"{slug}_responses.json").read_text(encoding="utf-8"))
    pairs = []
    for i, (p, n) in enumerate(zip(data["positive"], data["negative"])):
        if p.get("kept", False) and n.get("kept", False):
            pairs.append(Pair(i, p["question"], p["response"], n["response"]))
    return pairs


def split(pairs: list, seed: int = 0, frac: float = 0.8) -> tuple[list, list]:
    ordered = sorted(pairs, key=lambda p: p.index)
    idx = list(range(len(ordered)))
    random.Random(seed).shuffle(idx)
    n_train = round(frac * len(ordered))
    train = [ordered[i] for i in idx[:n_train]]
    test = [ordered[i] for i in idx[n_train:]]
    return train, test


def start_vector(src: Path, slug: str, layer: int, train_indices: list[int]) -> torch.Tensor:
    raw = torch.load(Path(src) / f"{slug}_raw.pt")
    kept = list(raw["kept_questions"])
    rows = []
    for i in train_indices:
        if i not in kept:
            raise ValueError(f"question {i} not in kept_questions of {slug}")
        rows.append(kept.index(i))
    pos = raw["positive"][layer]["residual"][rows].float()
    neg = raw["negative"][layer]["residual"][rows].float()
    return (pos.mean(0) - neg.mean(0)).float()


def full_source_vector(src: Path, slug: str, layer: int) -> torch.Tensor:
    return torch.load(Path(src) / f"{slug}.pt")[layer]


def supergpqa_prompts(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
        text = r["question"] + "\n\n" + opts + "\n\nWhich option is correct?"
        out.append({"id": r["id"], "question": text, "answer": r["answer"]})
    return out


def usable_traits(src: Path, slugs: list[str], min_pairs: int = 20) -> tuple[list[str], dict[str, str]]:
    kept: list[str] = []
    reasons: dict[str, str] = {}
    for slug in slugs:
        if not (Path(src) / f"{slug}_responses.json").exists():
            reasons[slug] = "missing"
            continue
        n = len(load_pairs(src, slug))
        if n < min_pairs:
            reasons[slug] = f"{n} kept pairs < {min_pairs}"
        else:
            kept.append(slug)
    return kept, reasons
