from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from persvectrain.data import Pair
from persvectrain.losses import kl_loss, persona_margin
from persvectrain.steering import answer_logprob, answer_logprobs, encode, steer
from persvectrain.train import RefAnswer, TrainConfig, TraitData, train

LAYER = 1
MAX_TOK = 12


def _pairs(n: int, offset: int = 0) -> list[Pair]:
    return [Pair(offset + i, f"q{offset + i}?", "a" * 10, "b" * 10) for i in range(n)]


def _trait(slug: str, n_train: int = 6, zero: bool = False, seed: int = 1) -> TraitData:
    g = torch.Generator().manual_seed(seed)
    start = torch.zeros(16) if zero else 0.1 * torch.randn(16, generator=g)
    return TraitData(slug=slug, train=_pairs(n_train), held=_pairs(2, offset=100), start=start,
                     source=start.clone())


def _refs(n: int = 3) -> list[RefAnswer]:
    return [RefAnswer(id=f"r{i}", question=f"Question {i}?", answer_text="The answer is C.", gold="C")
            for i in range(n)]


def _cfg(**kw) -> TrainConfig:
    base = dict(model="tiny", layer=LAYER, mode="solo", traits=["t1"], source=Path("src"), out=Path("out"),
                run_name="test", lambda_kl=0.0, epochs=3, lr=1e-2, batch=2, kl_batch=2,
                max_answer_tokens=MAX_TOK, init="dom", seed=0)
    base.update(kw)
    return TrainConfig(**base)


def _mean_margin(model, tok, pairs, vec) -> float:
    total = []
    with torch.no_grad():
        for p in pairs:
            ids_p, s_p = encode(tok, p.question, p.pos, MAX_TOK)
            ids_n, s_n = encode(tok, p.question, p.neg, MAX_TOK)
            lp0_p, lp0_n = answer_logprob(model, ids_p, s_p), answer_logprob(model, ids_n, s_n)
            for sign in (1, -1):
                with steer(model, LAYER, sign * vec):
                    lpc_p, lpc_n = answer_logprob(model, ids_p, s_p), answer_logprob(model, ids_n, s_n)
                total.append(float(persona_margin(lpc_p, lp0_p, lpc_n, lp0_n, sign)))
    return sum(total) / len(total)


def _mean_kl(model, tok, refs, vec) -> float:
    out = []
    with torch.no_grad():
        for r in refs:
            ids, s = encode(tok, r.question, r.answer_text, MAX_TOK)
            lp0 = answer_logprobs(model, ids, s)
            with steer(model, LAYER, vec):
                lpc = answer_logprobs(model, ids, s)
            out.append(float(kl_loss(lp0, lpc)))
    return sum(out) / len(out)


def test_solo_raises_train_margin(tiny_model, char_tok):
    td = _trait("t1", zero=True)
    vecs = train(tiny_model, char_tok, _cfg(init="dom"), {"t1": td}, _refs(), lambda d: None)
    assert set(vecs) == {"t1"}
    v = vecs["t1"]
    assert v.dtype == torch.float32 and v.device.type == "cpu" and v.shape == (16,)
    before = _mean_margin(tiny_model, char_tok, td.train, td.start)
    after = _mean_margin(tiny_model, char_tok, td.train, v)
    assert after > before


def test_kl_term_keeps_kl_lower(tiny_model, char_tok):
    td = _trait("t1")
    refs = _refs()
    v0 = train(tiny_model, char_tok, _cfg(lambda_kl=0.0), {"t1": td}, refs, lambda d: None)["t1"]
    v100 = train(tiny_model, char_tok, _cfg(lambda_kl=100.0), {"t1": td}, refs, lambda d: None)["t1"]
    assert _mean_kl(tiny_model, char_tok, refs, v100) < _mean_kl(tiny_model, char_tok, refs, v0)


def test_only_vector_changes(tiny_model, char_tok):
    before = {k: t.clone() for k, t in tiny_model.state_dict().items()}
    train(tiny_model, char_tok, _cfg(lambda_kl=1.0), {"t1": _trait("t1")}, _refs(), lambda d: None)
    after = tiny_model.state_dict()
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k]), k
    assert all(not p.requires_grad for p in tiny_model.parameters())


def test_log_step_fields(tiny_model, char_tok):
    traits = {"t1": _trait("t1", n_train=5), "t2": _trait("t2", n_train=3, seed=2)}
    logs: list[dict] = []
    cfg = _cfg(traits=["t1", "t2"], lambda_kl=1.0, epochs=2, batch=2)
    train(tiny_model, char_tok, cfg, traits, _refs(), logs.append)
    keys = {"step", "trait", "sign", "loss_p", "loss_kl", "norm", "lr"}
    for d in logs:
        assert keys <= set(d), d
        assert d["sign"] in (-1, 1) and isinstance(d["sign"], int)
        for k in ("loss_p", "loss_kl", "norm", "lr"):
            assert isinstance(d[k], float), k
        assert d["loss_kl"] > 0.0 or d["norm"] == 0.0
    assert [d["step"] for d in logs] == list(range(len(logs)))
    n1 = sum(d["trait"] == "t1" for d in logs)
    n2 = sum(d["trait"] == "t2" for d in logs)
    assert n1 == 2 * math.ceil(5 / 2) and n2 == 2 * math.ceil(3 / 2)


def test_lambda_kl_zero_logs_zero_kl(tiny_model, char_tok):
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(lambda_kl=0.0), {"t1": _trait("t1")}, _refs(), logs.append)
    assert logs and all(d["loss_kl"] == 0.0 for d in logs)


def test_zero_init(tiny_model, char_tok):
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(init="zero"), {"t1": _trait("t1")}, _refs(), logs.append)
    assert logs[0]["norm"] == 0.0
    assert logs[-1]["norm"] > 0.0


def test_dom_init_starts_at_start_vector(tiny_model, char_tok):
    td = _trait("t1")
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(init="dom"), {"t1": td}, _refs(), logs.append)
    assert logs[0]["norm"] == pytest.approx(float(td.start.norm()), rel=1e-6)


def test_same_seed_same_vectors(tiny_model, char_tok):
    td = _trait("t1")
    cfg = _cfg(lambda_kl=1.0)
    a = train(tiny_model, char_tok, cfg, {"t1": td}, _refs(), lambda d: None)["t1"]
    b = train(tiny_model, char_tok, cfg, {"t1": td}, _refs(), lambda d: None)["t1"]
    assert torch.equal(a, b)
    c = train(tiny_model, char_tok, _cfg(lambda_kl=1.0, seed=1), {"t1": td}, _refs(), lambda d: None)["t1"]
    assert not torch.equal(a, c)


def test_joint_not_implemented(tiny_model, char_tok):
    with pytest.raises(NotImplementedError):
        train(tiny_model, char_tok, _cfg(mode="joint"), {"t1": _trait("t1")}, _refs(), lambda d: None)
