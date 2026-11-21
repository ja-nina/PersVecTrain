from __future__ import annotations

import json
from pathlib import Path

import torch

from persvectrain.data import (
    full_source_vector,
    load_pairs,
    split,
    start_vector,
    supergpqa_prompts,
    usable_traits,
    Pair,
)

REAL = Path(__file__).resolve().parent.parent / "data" / "supergpqa_200_299.jsonl"


def _responses(tmp_path, slug, kept_flags):
    rec = lambda side, i, k: {"question": f"q{i}", "system_prompt": "s",
                              "response": f"{side}{i}", "kept": k}
    data = {
        "positive": [rec("pos", i, k) for i, k in enumerate(kept_flags)],
        "negative": [rec("neg", i, k) for i, k in enumerate(kept_flags)],
    }
    (tmp_path / f"{slug}_responses.json").write_text(json.dumps(data))


def test_load_pairs_only_kept(tmp_path):
    _responses(tmp_path, "t", [True, False, True])
    pairs = load_pairs(tmp_path, "t")
    assert [p.index for p in pairs] == [0, 2]
    assert pairs[1] == Pair(index=2, question="q2", pos="pos2", neg="neg2")


def test_split_deterministic_and_disjoint():
    pairs = [Pair(i, f"q{i}", "p", "n") for i in range(40)]
    tr, te = split(pairs, seed=0)
    assert (len(tr), len(te)) == (32, 8)
    assert not {p.index for p in tr} & {p.index for p in te}
    tr2, te2 = split(pairs, seed=0)
    assert [p.index for p in tr] == [p.index for p in tr2]
    assert [p.index for p in te] == [p.index for p in te2]
    tr3, _ = split(pairs, seed=1)
    assert [p.index for p in tr3] != [p.index for p in tr]


def test_start_vector_uses_kept_questions(tmp_path):
    pos = torch.tensor([[1.0, 0], [3, 0], [5, 0]]).half()
    neg = torch.tensor([[0.0, 1], [0, 1], [0, 3]]).half()
    raw = {
        "positive": {4: {"residual": pos}},
        "negative": {4: {"residual": neg}},
        "kept_questions": [0, 2, 5],
    }
    torch.save(raw, tmp_path / "t_raw.pt")
    v = start_vector(tmp_path, "t", 4, [2, 5])
    assert v.dtype == torch.float32
    assert torch.allclose(v, torch.tensor([4.0, -2.0]))
    try:
        start_vector(tmp_path, "t", 4, [1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_full_source_vector(tmp_path):
    torch.save({3: torch.arange(3.0)}, tmp_path / "t.pt")
    assert torch.equal(full_source_vector(tmp_path, "t", 3), torch.arange(3.0))


def test_supergpqa_format(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "Q?", "options": ["x", "y"],
                             "answer": "B"}) + "\n")
    out = supergpqa_prompts(p)
    assert out[0]["id"] == "a" and out[0]["answer"] == "B"
    assert out[0]["question"] == "Q?\n\nA. x\nB. y\n\nWhich option is correct?"
    assert set(out[0]) == {"id", "question", "answer"}
    assert len(supergpqa_prompts(REAL)) == 100


def test_usable_traits_drops_with_reason(tmp_path):
    _responses(tmp_path, "first", [True] * 25)
    _responses(tmp_path, "second", [True] * 5 + [False] * 20)
    kept, reasons = usable_traits(tmp_path, ["first", "second", "third"])
    assert kept == ["first"]
    assert reasons == {"second": "5 kept pairs < 20", "third": "missing"}
