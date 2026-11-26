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
                max_answer_tokens=MAX_TOK, kl_tokens=MAX_TOK, init="dom", seed=0)
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
    for d in _steps(logs):
        assert keys <= set(d), d
        assert d["sign"] in (-1, 1) and isinstance(d["sign"], int)
        for k in ("loss_p", "loss_kl", "norm", "lr"):
            assert isinstance(d[k], float), k
        assert d["loss_kl"] > 0.0 or d["norm"] == 0.0
    assert [d["step"] for d in _steps(logs)] == list(range(len(_steps(logs))))
    n1 = sum(d["trait"] == "t1" for d in _steps(logs))
    n2 = sum(d["trait"] == "t2" for d in _steps(logs))
    assert n1 == 2 * math.ceil(5 / 2) and n2 == 2 * math.ceil(3 / 2)


def test_lambda_kl_zero_logs_zero_kl(tiny_model, char_tok):
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(lambda_kl=0.0), {"t1": _trait("t1")}, _refs(), logs.append)
    assert _steps(logs) and all(d["loss_kl"] == 0.0 for d in _steps(logs))


def test_zero_init(tiny_model, char_tok):
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(init="zero"), {"t1": _trait("t1")}, _refs(), logs.append)
    assert _steps(logs)[0]["norm"] == 0.0
    assert _steps(logs)[-1]["norm"] > 0.0


def test_dom_init_starts_at_start_vector(tiny_model, char_tok):
    td = _trait("t1")
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(init="dom"), {"t1": td}, _refs(), logs.append)
    assert _steps(logs)[0]["norm"] == pytest.approx(float(td.start.norm()), rel=1e-6)


def test_same_seed_same_vectors(tiny_model, char_tok):
    td = _trait("t1")
    cfg = _cfg(lambda_kl=1.0, epochs=1)
    a = train(tiny_model, char_tok, cfg, {"t1": td}, _refs(), lambda d: None)["t1"]
    b = train(tiny_model, char_tok, cfg, {"t1": td}, _refs(), lambda d: None)["t1"]
    assert torch.equal(a, b)
    c = train(tiny_model, char_tok, _cfg(lambda_kl=1.0, epochs=1, seed=1), {"t1": td}, _refs(), lambda d: None)["t1"]
    assert not torch.equal(a, c)




@pytest.mark.parametrize("lambda_kl", [0.0, 3.0])
def test_step_gradient_is_mean_loss_gradient(tiny_model, char_tok, monkeypatch, lambda_kl):
    """The step's gradient on v = grad of mean persona loss over its pairs + lambda_kl * mean KL over its answers."""
    from persvectrain import train as train_mod
    from persvectrain.losses import persona_loss

    grads = []
    real_step = torch.optim.AdamW.step

    def recording_step(self, *a, **kw):
        grads.append(self.param_groups[0]["params"][0].grad.detach().clone())
        return real_step(self, *a, **kw)

    monkeypatch.setattr(train_mod.torch.optim.AdamW, "step", recording_step)
    td = _trait("t1", n_train=2)
    refs = _refs(3)
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(lambda_kl=lambda_kl, batch=2, kl_batch=2, epochs=1), {"t1": td}, refs,
          logs.append)
    assert len(grads) == 1
    sign = _steps(logs)[0]["sign"]

    v = td.start.clone().requires_grad_(True)
    p_losses = []
    for p in td.train:
        ids_p, s_p = encode(char_tok, p.question, p.pos, MAX_TOK)
        ids_n, s_n = encode(char_tok, p.question, p.neg, MAX_TOK)
        with torch.no_grad():
            lp0_p, lp0_n = answer_logprob(tiny_model, ids_p, s_p), answer_logprob(tiny_model, ids_n, s_n)
        with steer(tiny_model, LAYER, sign * v):
            lpc_p, lpc_n = answer_logprob(tiny_model, ids_p, s_p), answer_logprob(tiny_model, ids_n, s_n)
        p_losses.append(persona_loss(lpc_p, lp0_p, lpc_n, lp0_n, sign, 0.1))
    total = torch.stack(p_losses).mean()
    kls = []
    for r in refs[:2]:
        ids, s = encode(char_tok, r.question, r.answer_text, MAX_TOK)
        with torch.no_grad():
            lp0 = answer_logprobs(tiny_model, ids, s)
        with steer(tiny_model, LAYER, sign * v):
            kls.append(kl_loss(lp0, answer_logprobs(tiny_model, ids, s)))
    if lambda_kl:
        total = total + lambda_kl * torch.stack(kls).mean()
    (expected,) = torch.autograd.grad(total, v)
    assert torch.allclose(grads[0], expected, rtol=1e-4, atol=1e-7)
    assert _steps(logs)[0]["loss_p"] == pytest.approx(float(torch.stack(p_losses).mean().detach()), rel=1e-5)
    if lambda_kl:
        assert _steps(logs)[0]["loss_kl"] == pytest.approx(float(torch.stack(kls).mean().detach()), rel=1e-5)


def test_trait_order_does_not_change_vectors(tiny_model, char_tok):
    traits = {"A": _trait("A", seed=1), "B": _trait("B", n_train=4, seed=2)}
    ab = train(tiny_model, char_tok, _cfg(traits=["A", "B"], lambda_kl=1.0, epochs=1), traits, _refs(), lambda d: None)
    ba = train(tiny_model, char_tok, _cfg(traits=["B", "A"], lambda_kl=1.0, epochs=1), traits, _refs(), lambda d: None)
    assert torch.equal(ab["A"], ba["A"]) and torch.equal(ab["B"], ba["B"])


# ---------------------------------------------------------------- joint


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def test_solo_logs_single_kind(tiny_model, char_tok):
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(epochs=1), {"t1": _trait("t1")}, _refs(), logs.append)
    for d in _steps(logs):
        assert d["kind"] == "single" and d["weights"] == {"t1": d["sign"]} and d["loss_o"] == 0.0


def test_kl_batch_must_be_positive_when_kl_on(tiny_model, char_tok):
    with pytest.raises(ValueError):
        train(tiny_model, char_tok, _cfg(lambda_kl=1.0, kl_batch=0), {"t1": _trait("t1")}, _refs(), lambda d: None)
    # KL off: kl_batch is unused
    train(tiny_model, char_tok, _cfg(lambda_kl=0.0, kl_batch=0, epochs=1), {"t1": _trait("t1")}, _refs(),
          lambda d: None)


def test_solo_equals_joint_without_extras(tiny_model, char_tok):
    td = _trait("t1", n_train=5)
    solo_logs: list[dict] = []
    joint_logs: list[dict] = []
    solo = train(tiny_model, char_tok, _cfg(lambda_kl=1.0, epochs=2), {"t1": td}, _refs(), solo_logs.append)
    joint = train(tiny_model, char_tok, _cfg(mode="joint", lambda_kl=1.0, epochs=2, lambda_o=0.0, mix_every=0),
                  {"t1": td}, _refs(), joint_logs.append)
    assert torch.equal(solo["t1"], joint["t1"])
    assert solo_logs == joint_logs


def test_joint_overlap_reduces_cosine(tiny_model, char_tok):
    g = torch.Generator().manual_seed(3)
    a = 0.3 * torch.randn(16, generator=g)
    noise = torch.randn(16, generator=g)
    noise = noise - (noise @ a) / (a @ a) * a
    b = 0.9 * a / a.norm() + math.sqrt(1 - 0.81) * noise / noise.norm()
    b = b * a.norm()
    assert _cos(a, b) == pytest.approx(0.9, abs=1e-5)
    traits = {"A": TraitData("A", _pairs(4), _pairs(2, 100), a, a.clone()),
              "B": TraitData("B", _pairs(4), _pairs(2, 100), b, b.clone())}
    logs: list[dict] = []
    out = train(tiny_model, char_tok, _cfg(mode="joint", traits=["A", "B"], lambda_o=10.0, lambda_kl=0.0,
                                           mix_every=0, epochs=3), traits, _refs(), logs.append)
    assert abs(_cos(out["A"], out["B"])) < 0.9
    assert _steps(logs)[0]["loss_o"] == pytest.approx(0.81, rel=1e-4)


def test_joint_round_robin_and_mix_schedule(tiny_model, char_tok):
    traits = {"A": _trait("A", n_train=4, seed=1), "B": _trait("B", n_train=2, seed=2)}
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(mode="joint", traits=["A", "B"], mix_every=3, epochs=1, batch=2),
          traits, _refs(), logs.append)
    # singles per epoch: A, B, A (round robin); mix at s % 3 == 2; ends when singles are used up
    assert [d["kind"] for d in _steps(logs)] == ["single", "single", "mix", "single"]
    assert [d["trait"] for d in _steps(logs) if d["kind"] == "single"] == ["A", "B", "A"]
    assert [d["step"] for d in _steps(logs)] == [0, 1, 2, 3]


def test_mix_weights_bounded(tiny_model, char_tok):
    traits = {s: _trait(s, n_train=4, seed=i + 1) for i, s in enumerate(["A", "B", "C"])}
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(mode="joint", traits=["A", "B", "C"], mix_every=2, epochs=2, lambda_o=1.0),
          traits, _refs(), logs.append)
    mixes = [d for d in _steps(logs) if d["kind"] == "mix"]
    assert mixes
    for d in mixes:
        assert len(d["weights"]) == 2 and d["sign"] == 0
        assert d["trait"] == "+".join(d["weights"])
        assert set(d["norms"]) == set(d["weights"])
        for w in d["weights"].values():
            assert isinstance(w, float) and 0.25 <= abs(w) <= 1.0
        assert d["norm"] == d["norms"][next(iter(d["weights"]))]
    for d in _steps(logs):
        assert d["loss_o"] > 0.0


def test_mix_step_reference_is_other_traits(tiny_model, char_tok, monkeypatch):
    from persvectrain import train as train_mod

    real_steer = train_mod.steer
    calls: list[tuple] = []

    def recording_steer(model, layer, vec):
        calls.append((None if vec is None else vec.detach().clone(), torch.is_grad_enabled()))
        return real_steer(model, layer, vec)

    monkeypatch.setattr(train_mod, "steer", recording_steer)
    traits = {"A": _trait("A", n_train=2, seed=1), "B": _trait("B", n_train=2, seed=2)}
    logs: list[dict] = []
    # one single batch per trait, all steps mix: 2 steps; inspect the first
    train(tiny_model, char_tok, _cfg(mode="joint", traits=["A", "B"], mix_every=1, epochs=1, batch=2,
                                     lambda_kl=1.0, kl_batch=2), traits, _refs(), logs.append)
    assert _steps(logs)[0]["kind"] == "mix"
    (sa, wa), (sb, wb) = _steps(logs)[0]["weights"].items()
    va, vb = traits[sa].start, traits[sb].start
    mix = wa * va + wb * vb
    n_start = 2 * 2 * 2  # start line: 2 traits x 2 pairs x c = +-1, no grad
    assert all(not g for _v, g in calls[:n_start])
    first = calls[n_start:n_start + 2 * 2 + 2 * 2 + 2]  # A: 2 refs + 2 steered, B: 2 refs + 2 steered, 2 KL
    exp = ([(wb * vb, False)] * 2 + [(mix, True)] * 2 + [(wa * va, False)] * 2 + [(mix, True)] * 2
           + [(mix, True)] * 2)
    assert len(first) == len(exp)
    for (vec, grad_on), (want, want_grad) in zip(first, exp):
        assert grad_on == want_grad
        assert torch.allclose(vec, want, atol=1e-6)


def test_mix_step_gradient(tiny_model, char_tok, monkeypatch):
    """Mix-step gradient = grad of mean persona over both traits' pairs (other vector detached, reference under
    the other trait's term) + lambda_kl * mean KL under the full mix (gradient to both)."""
    from persvectrain import train as train_mod
    from persvectrain.losses import persona_loss

    grads = []
    real_step = torch.optim.AdamW.step

    def recording_step(self, *a, **kw):
        grads.append([p.grad.detach().clone() for p in self.param_groups[0]["params"]])
        return real_step(self, *a, **kw)

    monkeypatch.setattr(train_mod.torch.optim.AdamW, "step", recording_step)
    traits = {"A": _trait("A", n_train=2, seed=1), "B": _trait("B", n_train=2, seed=2)}
    refs = _refs(3)
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(mode="joint", traits=["A", "B"], mix_every=1, epochs=1, batch=2,
                                     lambda_kl=2.0, kl_batch=2), traits, refs, logs.append)
    w = _steps(logs)[0]["weights"]
    v = {s: traits[s].start.clone().requires_grad_(True) for s in ("A", "B")}
    p_losses = []
    for s in w:
        (o,) = [x for x in w if x != s]
        other = (w[o] * v[o]).detach()
        sign = 1 if w[s] > 0 else -1
        for p in traits[s].train:  # batch == n_train: every pair is sampled
            ids_p, s_p = encode(char_tok, p.question, p.pos, MAX_TOK)
            ids_n, s_n = encode(char_tok, p.question, p.neg, MAX_TOK)
            with torch.no_grad(), steer(tiny_model, LAYER, other):
                r_p, r_n = answer_logprob(tiny_model, ids_p, s_p), answer_logprob(tiny_model, ids_n, s_n)
            with steer(tiny_model, LAYER, w[s] * v[s] + other):
                c_p, c_n = answer_logprob(tiny_model, ids_p, s_p), answer_logprob(tiny_model, ids_n, s_n)
            p_losses.append(persona_loss(c_p, r_p, c_n, r_n, sign, 0.1))
    kls = []
    for r in refs[:2]:
        ids, st = encode(char_tok, r.question, r.answer_text, MAX_TOK)
        with torch.no_grad():
            lp0 = answer_logprobs(tiny_model, ids, st)
        with steer(tiny_model, LAYER, w["A"] * v["A"] + w["B"] * v["B"]):
            kls.append(kl_loss(lp0, answer_logprobs(tiny_model, ids, st)))
    total = torch.stack(p_losses).mean() + 2.0 * torch.stack(kls).mean()
    ga, gb = torch.autograd.grad(total, [v["A"], v["B"]])
    assert torch.allclose(grads[0][0], ga, rtol=1e-4, atol=1e-7)
    assert torch.allclose(grads[0][1], gb, rtol=1e-4, atol=1e-7)
    assert _steps(logs)[0]["loss_p"] == pytest.approx(float(torch.stack(p_losses).mean().detach()), rel=1e-5)


# ---------------------------------------------------------------- saturation diagnostics


def _steps(logs: list[dict]) -> list[dict]:
    return [d for d in logs if d["kind"] != "start"]


def _start_margin(model, tok, pairs, vec, sign, norm="sum") -> float:
    out = []
    with torch.no_grad():
        for p in pairs:
            ids_p, s_p = encode(tok, p.question, p.pos, MAX_TOK)
            ids_n, s_n = encode(tok, p.question, p.neg, MAX_TOK)
            lp0 = answer_logprob(model, ids_p, s_p, norm), answer_logprob(model, ids_n, s_n, norm)
            with steer(model, LAYER, sign * vec):
                lpc = answer_logprob(model, ids_p, s_p, norm), answer_logprob(model, ids_n, s_n, norm)
            out.append(float(persona_margin(lpc[0], lp0[0], lpc[1], lp0[1], sign)))
    return sum(out) / len(out)


def test_solo_start_line_per_trait(tiny_model, char_tok):
    traits = {"t1": _trait("t1", n_train=4), "t2": _trait("t2", n_train=3, seed=2)}
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(traits=["t1", "t2"], epochs=1, beta=0.5), traits, _refs(), logs.append)
    starts = [(i, d) for i, d in enumerate(logs) if d["kind"] == "start"]
    assert [i for i, _ in starts] == [0, 1 + math.ceil(4 / 2)]
    for (_, d), slug in zip(starts, ("t1", "t2")):
        assert set(d["traits"]) == {slug} and d["beta"] == 0.5
        st = d["traits"][slug]
        for sign, key in ((1, "pos"), (-1, "neg")):
            want = _start_margin(tiny_model, char_tok, traits[slug].train, traits[slug].start, sign)
            assert st[f"margin_{key}"] == pytest.approx(want, rel=1e-5, abs=1e-6)
            assert st[f"beta_margin_{key}"] == pytest.approx(0.5 * want, rel=1e-5, abs=1e-6)
    assert starts[1][1]["step"] == math.ceil(4 / 2)  # the next step's index


def test_joint_start_line_holds_every_trait(tiny_model, char_tok):
    traits = {"A": _trait("A", n_train=2, seed=1), "B": _trait("B", n_train=2, seed=2)}
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(mode="joint", traits=["A", "B"], mix_every=2, epochs=1), traits, _refs(),
          logs.append)
    assert logs[0]["kind"] == "start" and set(logs[0]["traits"]) == {"A", "B"}
    assert sum(d["kind"] == "start" for d in logs) == 1


def test_steps_log_margin_and_cos_start(tiny_model, char_tok):
    traits = {"A": _trait("A", n_train=4, seed=1), "B": _trait("B", n_train=4, seed=2)}
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(mode="joint", traits=["A", "B"], mix_every=2, epochs=2), traits, _refs(),
          logs.append)
    steps = _steps(logs)
    assert {d["kind"] for d in steps} == {"single", "mix"}
    for d in steps:
        assert isinstance(d["margin"], float)
        assert isinstance(d["cos_start"], float) and -1.0 - 1e-6 <= d["cos_start"] <= 1.0 + 1e-6
    assert steps[0]["cos_start"] == pytest.approx(1.0, abs=1e-6)  # dom init: v starts at its start vector
    mix = next(d for d in steps if d["kind"] == "mix")
    assert set(mix["cos_starts"]) == set(mix["weights"])

    zlogs: list[dict] = []
    train(tiny_model, char_tok, _cfg(init="zero", epochs=1), {"t1": _trait("t1")}, _refs(), zlogs.append)
    assert all(d["cos_start"] is None for d in _steps(zlogs))
    assert zlogs[0]["traits"]["t1"]["margin_pos"] == 0.0 and zlogs[0]["traits"]["t1"]["margin_neg"] == 0.0


def test_step_margin_is_mean_pair_margin(tiny_model, char_tok):
    td = _trait("t1", n_train=2)
    logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(batch=2, epochs=1), {"t1": td}, _refs(), logs.append)
    (d,) = _steps(logs)
    want = _start_margin(tiny_model, char_tok, td.train, td.start, d["sign"])
    assert d["margin"] == pytest.approx(want, rel=1e-5, abs=1e-6)


def test_logprob_norm_mean_divides_by_answer_tokens(tiny_model, char_tok):
    td = _trait("t1", n_train=2)  # answers "a" * 10 / "b" * 10: 10 tokens each
    s_logs: list[dict] = []
    m_logs: list[dict] = []
    train(tiny_model, char_tok, _cfg(batch=2, epochs=1), {"t1": td}, _refs(), s_logs.append)
    train(tiny_model, char_tok, _cfg(batch=2, epochs=1, logprob_norm="mean"), {"t1": td}, _refs(), m_logs.append)
    for key in ("margin_pos", "margin_neg"):
        assert m_logs[0]["traits"]["t1"][key] == pytest.approx(s_logs[0]["traits"]["t1"][key] / 10, rel=1e-5)
    assert _steps(m_logs)[0]["margin"] == pytest.approx(_steps(s_logs)[0]["margin"] / 10, rel=1e-5)
    with pytest.raises(ValueError):
        train(tiny_model, char_tok, _cfg(logprob_norm="max"), {"t1": td}, _refs(), lambda d: None)


def test_kl_tokens_truncates_kl_answers_not_persona_pairs():
    from persvectrain.train import _encode_kl
    from persvectrain.steering import encode as enc
    tok = char_tok_for_test()
    refs = [RefAnswer(id="r0", question="Q?", answer_text="x" * 40, gold="C")]
    cfg = _cfg(kl_tokens=30, max_answer_tokens=12, lambda_kl=1.0)
    ids, start = _encode_kl(tok, cfg, refs)[0]
    assert ids.shape[1] - start == 30
    assert enc(tok, "Q?", "x" * 40, cfg.max_answer_tokens)[0].shape[1] - start == 12  # persona pairs keep 12


def char_tok_for_test():
    from tests.conftest import CharTokenizer
    return CharTokenizer()


@pytest.mark.parametrize("mode", ["solo", "joint"])
def test_fix_norm_keeps_the_start_length_and_changes_the_direction(tiny_model, char_tok, mode):
    tds = {"t1": _trait("t1"), "t2": _trait("t2")}
    cfg = _cfg(mode=mode, traits=["t1", "t2"], lambda_kl=1.0, fix_norm=1)
    vecs = train(tiny_model, char_tok, cfg, tds, _refs(), lambda d: None)
    free = train(tiny_model, char_tok, _cfg(mode=mode, traits=["t1", "t2"], lambda_kl=1.0), tds, _refs(),
                 lambda d: None)
    for s, td in tds.items():
        assert float(vecs[s].norm()) == pytest.approx(float(td.start.norm()), rel=1e-5)
        assert not torch.allclose(vecs[s] / vecs[s].norm(), td.start / td.start.norm(), atol=1e-4)
        assert abs(float(free[s].norm()) - float(td.start.norm())) > 1e-4  # without it the length moves


def test_fix_norm_needs_dom_init(tiny_model, char_tok):
    with pytest.raises(ValueError, match="fix_norm"):
        train(tiny_model, char_tok, _cfg(init="zero", fix_norm=1), {"t1": _trait("t1")}, _refs(), lambda d: None)


@pytest.mark.parametrize("mode", ["solo", "joint"])
def test_grad_checkpoint_gives_the_same_vectors(tiny_model, char_tok, mode):
    tds = {"t1": _trait("t1"), "t2": _trait("t2")}
    kw = dict(mode=mode, traits=["t1", "t2"], lambda_kl=1.0)
    plain = train(tiny_model, char_tok, _cfg(**kw), tds, _refs(), lambda d: None)
    ckpt = train(tiny_model, char_tok, _cfg(**kw, grad_checkpoint=1), tds, _refs(), lambda d: None)
    for s in tds:
        assert torch.allclose(ckpt[s], plain[s], rtol=1e-5, atol=1e-6)


def test_checkpoint_layers_wraps_only_later_layers_and_restores(tiny_model):
    from persvecfast.smoke import decoder_layers
    from persvectrain.steering import checkpoint_layers
    layers = decoder_layers(tiny_model)
    before = [l.__dict__.get("forward") for l in layers]
    with checkpoint_layers(tiny_model, LAYER):
        assert [("forward" in l.__dict__) for l in layers] == [i > LAYER for i in range(len(layers))]
    assert [l.__dict__.get("forward") for l in layers] == before
    with checkpoint_layers(tiny_model, LAYER, on=False):
        assert not any("forward" in l.__dict__ for l in layers)
