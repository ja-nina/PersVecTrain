from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from persvecfast.extract import RANDOM_CONTROL_SEED, random_control
from persvecfast.smoke import decoder_layers

from persvectrain import evaluate as ev
from persvectrain.data import Pair
from persvectrain.generate import count_new_tokens
from persvectrain.losses import persona_margin
from persvectrain.steering import answer_logprob, encode, steer
from persvectrain.train import RefAnswer, TrainConfig, TraitData

LAYER = 1
MAX_TOK = 12
CS = ("c=-1", "c=-0.5", "c=+0.5", "c=+1")


# ---------------------------------------------------------------- pass_rule (pure)


def _kind(margins, kls) -> dict:
    return {c: {"margins": list(margins), "kls": list(kls)} for c in CS}


def _trait_eval(trained_m, source_m, trained_kl, source_kl, random_kl, tokens=None) -> dict:
    """tokens: {"unsteered": u, "trained": t, "source_full": s, "trained_random": r}, each a per-prompt token
    list (a number: that many tokens on every prompt), or None for no free generation."""
    kinds = {"trained": _kind(trained_m, trained_kl), "source_full": _kind(source_m, source_kl),
             "trained_random": _kind([0.0] * len(trained_m), random_kl)}
    te = {"kinds": kinds}
    if tokens is not None:
        n = len(trained_m)
        tok = {k: (list(v) if isinstance(v, (list, tuple)) else [v] * n) for k, v in tokens.items()}
        fg = {"unsteered": {"tokens": tok["unsteered"]}}
        for k in ("trained", "source_full", "trained_random"):
            fg[k] = {c: {"tokens": tok[k]} for c in ("c=+1", "c=-1")}
        te["free_gen"] = fg
    return te


GOOD_TOKENS = {"unsteered": 100.0, "trained": 105.0, "source_full": 150.0, "trained_random": 110.0}


def test_pass_rule_all_better_passes():
    te = _trait_eval([2, 3, 4, 5], [1, 2, 3, 4], [0.1, 0.2, 0.1, 0.2], [1.0, 1.1, 0.9, 1.0],
                     [0.5, 0.6, 0.5, 0.6], GOOD_TOKENS)
    r = ev.pass_rule(te)
    assert r["passed"] is True and r["failed"] == [] and r["skipped"] == []
    assert set(r["checks"]) == {"c=+1", "c=-1"}
    assert set(r["checks"]["c=+1"]) == {"persona", "kl", "length", "random"}
    assert all(chk["passed"] for c in r["checks"].values() for chk in c.values())


def test_pass_rule_persona_lower_by_more_than_one_se_fails():
    # paired differences -1, -1.1, -0.9, -1: mean -1, SE about 0.04
    te = _trait_eval([0, 0.9, 2.1, 3], [1, 2, 3, 4], [0.1] * 4, [1.0] * 4, [0.5] * 4, GOOD_TOKENS)
    r = ev.pass_rule(te)
    assert r["passed"] is False
    assert r["failed"] == ["persona@c=+1", "persona@c=-1"]
    chk = r["checks"]["c=+1"]["persona"]
    assert chk["diff_mean"] == pytest.approx(-1.0)
    assert chk["diff_se"] == pytest.approx(math.sqrt(0.02 / 3 / 4))


def test_pass_rule_persona_within_one_se_passes():
    # paired differences -1, +1, -0.8, +0.6: mean -0.05, SE about 0.47
    te = _trait_eval([0, 3, 2.2, 4.6], [1, 2, 3, 4], [0.1] * 4, [1.0] * 4, [0.5] * 4, GOOD_TOKENS)
    r = ev.pass_rule(te)
    assert r["checks"]["c=+1"]["persona"]["passed"] is True
    assert r["passed"] is True


def test_pass_rule_kl_tie_within_one_se_fails():
    # trained KL - source KL: +0.1, -0.1, +0.1, -0.15: trained not lower beyond 1 SE
    te = _trait_eval([2] * 4, [1] * 4, [1.1, 0.9, 1.1, 0.85], [1.0] * 4, [2.0] * 4, GOOD_TOKENS)
    r = ev.pass_rule(te)
    assert r["failed"] == ["kl@c=+1", "kl@c=-1"]


def test_pass_rule_kl_worse_than_random_twin_fails_random():
    te = _trait_eval([2] * 4, [1] * 4, [0.5, 0.6, 0.5, 0.6], [1.0, 1.1, 1.0, 1.1], [0.1, 0.2, 0.1, 0.2],
                     GOOD_TOKENS)
    r = ev.pass_rule(te)
    assert r["passed"] is False
    assert r["failed"] == ["random@c=+1", "random@c=-1"]


def test_pass_rule_length_and_random_length():
    tokens = dict(GOOD_TOKENS, trained=170.0)  # further from unsteered than source (150) and the twin (110)
    te = _trait_eval([2] * 4, [1] * 4, [0.1] * 4, [1.0] * 4, [0.5] * 4, tokens)
    r = ev.pass_rule(te)
    assert r["failed"] == ["length@c=+1", "random@c=+1", "length@c=-1", "random@c=-1"]


def test_pass_rule_length_is_paired_per_prompt():
    # per prompt |trained - unsteered| = 2, 0, 4, 2 and |source - unsteered| = 12, 10, 14, 12: diff +10 each
    tokens = {"unsteered": [100, 90, 80, 70], "trained": [102, 90, 76, 72], "source_full": [112, 100, 66, 82],
              "trained_random": [102, 90, 76, 72]}
    te = _trait_eval([2] * 4, [1] * 4, [0.1] * 4, [1.0] * 4, [0.5] * 4, tokens)
    r = ev.pass_rule(te)
    ch = r["checks"]["c=+1"]
    assert ch["length"]["diff_mean"] == pytest.approx(10.0) and ch["length"]["diff_se"] == pytest.approx(0.0)
    assert ch["length"]["passed"] is True
    assert ch["random"]["length_diff_mean"] == pytest.approx(0.0) and ch["random"]["length_passed"] is True
    assert r["passed"] is True


def test_pass_rule_length_tie_within_one_se_fails_source_check():
    # shift_source - shift_trained per prompt: +2, -2, +1, 0: mean 0.25 > 0 but within 1 SE (about 0.85)
    tokens = {"unsteered": [100] * 4, "trained": [110, 110, 110, 110], "source_full": [112, 108, 111, 110],
              "trained_random": [150] * 4}
    te = _trait_eval([2] * 4, [1] * 4, [0.1] * 4, [1.0] * 4, [0.5] * 4, tokens)
    r = ev.pass_rule(te)
    ch = r["checks"]["c=+1"]["length"]
    assert 0 < ch["diff_mean"] <= ch["diff_se"]
    assert ch["passed"] is False
    assert r["failed"] == ["length@c=+1", "length@c=-1"]


def test_pass_rule_random_length_tie_within_one_se_passes():
    # shift_random - shift_trained per prompt: -2, +2, -1, 0: mean -0.25, within 1 SE: no worse than the twin
    tokens = {"unsteered": [100] * 4, "trained": [110] * 4, "source_full": [200] * 4,
              "trained_random": [108, 112, 109, 110]}
    te = _trait_eval([2] * 4, [1] * 4, [0.1] * 4, [1.0] * 4, [0.5] * 4, tokens)
    r = ev.pass_rule(te)
    rnd = r["checks"]["c=+1"]["random"]
    assert -rnd["length_diff_se"] <= rnd["length_diff_mean"] < 0
    assert rnd["length_passed"] is True and rnd["passed"] is True and r["passed"] is True

    # beyond 1 SE: the trained vector moves length more than its twin
    tokens["trained_random"] = [101, 102, 100, 101]
    r = ev.pass_rule(_trait_eval([2] * 4, [1] * 4, [0.1] * 4, [1.0] * 4, [0.5] * 4, tokens))
    assert r["failed"] == ["random@c=+1", "random@c=-1"]


def test_pass_rule_length_needs_paired_tokens():
    tokens = {"unsteered": [100] * 4, "trained": [110] * 3, "source_full": [200] * 4, "trained_random": [150] * 4}
    with pytest.raises(ValueError):
        ev.pass_rule(_trait_eval([2] * 4, [1] * 4, [0.1] * 4, [1.0] * 4, [0.5] * 4, tokens))


def test_pass_rule_without_free_gen_skips_length():
    te = _trait_eval([2] * 4, [1] * 4, [0.1] * 4, [1.0] * 4, [0.5] * 4, None)
    r = ev.pass_rule(te)
    assert r["passed"] is True
    assert r["skipped"] == ["length@c=+1", "random-length@c=+1", "length@c=-1", "random-length@c=-1"]
    assert "length" not in r["checks"]["c=+1"]


def test_mean_se():
    assert ev.mean_se([1.0, 3.0]) == (2.0, pytest.approx(1.0))
    assert ev.mean_se([5.0]) == (5.0, None)
    assert ev.mean_se([]) == (None, None)


# ---------------------------------------------------------------- answer parsing / free generation


@pytest.mark.parametrize("text,letter", [
    ("... so the answer is C", "C"),
    ("Answer: **B**.", "B"),
    ("A first guess. The answer is (D), not A.", "D"),
    ("The answer is C. Re-checking, the final ANSWER: E", "E"),
    ("The answer is a mystery", None),
    ("I pick C", None),
])
def test_parse_answer(text, letter):
    assert ev.parse_answer(text) == letter


def _fake_gen(texts, counts, seen):
    def fake(model, tok, prompts, batch_size, max_new_tokens, **kw):
        hooked = bool(decoder_layers(model)[LAYER]._forward_hooks)
        seen.append({"prompts": list(prompts), "batch_size": batch_size, "max_new_tokens": max_new_tokens,
                     "hooked": hooked, "sampling": {k: v for k, v in kw.items() if k != "label"},
                     "rand": float(torch.rand(()))})
        n = len(prompts)
        return (texts * n)[:n], (counts * n)[:n]
    return fake


def test_free_generation_scores_with_fake_generate(tiny_model, char_tok, monkeypatch):
    seen = []
    monkeypatch.setattr(ev, "generate_counted", _fake_gen(["... answer is C"], [10, 30], seen))
    refs = _refs(4)
    r = ev.free_generation(tiny_model, char_tok, torch.ones(16), refs, seed=7, layer=LAYER, max_new_tokens=30,
                           batch_size=3, sampling={"temperature": 0.7})
    assert r["accuracy"] == 1.0 and r["accuracy_se"] == 0.0 and r["n"] == 4
    assert r["mean_tokens"] == 20.0 and r["cut_share"] == 0.5
    assert seen[0]["hooked"] is True and seen[0]["batch_size"] == 3 and seen[0]["max_new_tokens"] == 30
    assert seen[0]["sampling"] == {"temperature": 0.7}
    assert seen[0]["prompts"][0].startswith("<s>") and "Question 0?" in seen[0]["prompts"][0]
    assert not decoder_layers(tiny_model)[LAYER]._forward_hooks  # hook removed

    seen2 = []
    monkeypatch.setattr(ev, "generate_counted", _fake_gen(["answer is C", "answer is B"], [5], seen2))
    r = ev.free_generation(tiny_model, char_tok, None, refs, seed=7, max_new_tokens=30)
    assert r["accuracy"] == 0.5 and r["accuracy_se"] == pytest.approx(math.sqrt(1 / 3) / 2)
    assert seen2[0]["hooked"] is False and seen2[0]["rand"] == seen[0]["rand"]  # seeded the same


def test_count_new_tokens():
    new = torch.tensor([[5, 6, 0, 0], [5, 6, 7, 8], [9, 9, 9, 2]])
    assert count_new_tokens(new, {0, 2}) == [2, 4, 3]


# ---------------------------------------------------------------- evaluate on the tiny model


def _pairs(n: int, offset: int = 0, pos: str = "a" * 6, neg: str = "b" * 6) -> list[Pair]:
    return [Pair(offset + i, f"q{offset + i}?", pos, neg) for i in range(n)]


def _trait(slug: str, seed: int) -> TraitData:
    g = torch.Generator().manual_seed(seed)
    source = 0.5 * torch.randn(16, generator=g)
    return TraitData(slug=slug, train=_pairs(4), held=_pairs(3, offset=100), start=0.9 * source,
                     source=source)


def _refs(n: int = 3) -> list[RefAnswer]:
    return [RefAnswer(id=f"r{i}", question=f"Question {i}?", answer_text="The answer is C.", gold="C")
            for i in range(n)]


def _cfg(tmp_path: Path, **kw) -> TrainConfig:
    base = dict(model="qwen3.5-2b", layer=LAYER, mode="solo", traits=["t1"], source=tmp_path / "src",
                out=tmp_path / "out", run_name="run", max_answer_tokens=MAX_TOK, seed=0)
    base.update(kw)
    return TrainConfig(**base)


KINDS = ("trained", "source_full", "source_train", "source_at_trained_norm", "trained_random", "source_random")


def test_evaluate_tiny_every_kind_and_c(tiny_model, char_tok, tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(ev, "generate_counted", _fake_gen(["the answer is C"], [8], seen))
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    trained = {"t1": 2.0 * td.source + 0.1}
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, trained, _refs(), free_gen_tokens=16)

    t = res["traits"]["t1"]
    assert set(t["kinds"]) == set(KINDS)
    for kind in KINDS:
        assert set(t["kinds"][kind]) == set(CS)
        for c in CS:
            row = t["kinds"][kind][c]
            assert len(row["margins"]) == 3 and len(row["kls"]) == 3
            for key in ("margin_mean", "margin_se", "kl_mean", "kl_se"):
                assert isinstance(row[key], float), (kind, c, key)

    # margin of source_full at c = +1 on the first held pair, by hand
    p = td.held[0]
    ids_p, s_p = encode(char_tok, p.question, p.pos, MAX_TOK)
    ids_n, s_n = encode(char_tok, p.question, p.neg, MAX_TOK)
    with torch.no_grad():
        lp0 = answer_logprob(tiny_model, ids_p, s_p), answer_logprob(tiny_model, ids_n, s_n)
        with steer(tiny_model, LAYER, -0.5 * td.source):
            lpc = answer_logprob(tiny_model, ids_p, s_p), answer_logprob(tiny_model, ids_n, s_n)
    want = float(persona_margin(lpc[0], lp0[0], lpc[1], lp0[1], -1))
    assert t["kinds"]["source_full"]["c=-0.5"]["margins"][0] == pytest.approx(want, rel=1e-5, abs=1e-6)

    tn = float(trained["t1"].norm())
    assert t["norms"]["trained"] == pytest.approx(tn)
    assert t["norms"]["source_at_trained_norm"] == pytest.approx(tn)
    assert t["norms"]["trained_random"] == pytest.approx(tn)
    assert t["norms"]["source_full"] == pytest.approx(float(td.source.norm()))
    assert t["cos_with_source"]["source_full"] == pytest.approx(1.0)
    assert t["cos_with_source"]["source_at_trained_norm"] == pytest.approx(1.0)
    assert t["cos_with_source"]["source_train"] == pytest.approx(1.0)

    # free generation: unsteered once, then trained / source_full / trained_random at c = +-1
    assert len(seen) == 7 and seen[0]["hooked"] is False and all(s["hooked"] for s in seen[1:])
    assert all(s["max_new_tokens"] == 16 for s in seen)
    assert seen[0]["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    fg = t["free_gen"]
    assert fg["unsteered"]["accuracy"] == 1.0 and fg["unsteered"]["mean_tokens"] == 8.0
    for k in ("trained", "source_full", "trained_random"):
        assert set(fg[k]) == {"c=+1", "c=-1"}
    assert set(t["pass"]) >= {"checks", "failed", "passed", "skipped"}
    assert res["summary"] == {"t1": t["pass"]["passed"]}
    assert "joint" not in res

    saved = json.loads((tmp_path / "out" / "trained" / "run" / "eval.json").read_text(encoding="utf-8"))
    assert saved["traits"]["t1"]["kinds"]["trained"]["c=+1"]["margins"] == t["kinds"]["trained"]["c=+1"]["margins"]
    text = capsys.readouterr().out
    assert "t1" in text and "trained_random" in text and "pass t1:" in text


def test_evaluate_random_twin_matches_write_vectors(tiny_model, char_tok, tmp_path):
    cfg = _cfg(tmp_path, seed=3)
    td = _trait("t1", 1)
    v = td.source * 1.5
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": v}, _refs(2), free_gen=False)
    twin = random_control(v, RANDOM_CONTROL_SEED + cfg.layer)  # 0 + layer whatever cfg.seed is (PersVecFast / MA)
    want = float(torch.nn.functional.cosine_similarity(twin, td.source, dim=0))
    assert res["traits"]["t1"]["cos_with_source"]["trained_random"] == pytest.approx(want, abs=1e-6)
    src_twin = random_control(td.source, RANDOM_CONTROL_SEED + cfg.layer)
    want = float(torch.nn.functional.cosine_similarity(src_twin, td.source, dim=0))
    assert res["traits"]["t1"]["cos_with_source"]["source_random"] == pytest.approx(want, abs=1e-6)
    p = res["traits"]["t1"]["pass"]
    assert "free_gen" not in res["traits"]["t1"]
    assert "length@c=+1" in p["skipped"] and res["free_gen"]["enabled"] is False


def test_evaluate_joint_cos_matrix_and_mixes(tiny_model, char_tok, tmp_path):
    cfg = _cfg(tmp_path, mode="joint", traits=["t1", "t2"])
    tds = {"t1": _trait("t1", 1), "t2": _trait("t2", 2)}
    trained = {s: tds[s].source.clone() for s in tds}
    res = ev.evaluate(tiny_model, char_tok, cfg, tds, trained, _refs(2), free_gen=False)
    j = res["joint"]
    assert j["cos_matrix"]["t1"]["t1"] == pytest.approx(1.0)
    want = float(torch.nn.functional.cosine_similarity(trained["t1"], trained["t2"], dim=0))
    assert j["cos_matrix"]["t1"]["t2"] == pytest.approx(want) and j["cos_matrix"]["t2"]["t1"] == pytest.approx(want)
    assert len(j["mixes"]) == 4
    assert {tuple(m["weights"][s] for s in ("t1", "t2")) for m in j["mixes"]} == {(1, 1), (1, -1), (-1, 1),
                                                                                   (-1, -1)}
    for m in j["mixes"]:
        assert set(m["mix_margin"]) == {"t1", "t2"} and set(m["alone_margin"]) == {"t1", "t2"}
        for s in ("t1", "t2"):
            c = "c=+1" if m["weights"][s] > 0 else "c=-1"
            assert m["alone_margin"][s] == pytest.approx(res["traits"][s]["kinds"]["trained"][c]["margin_mean"])
            assert isinstance(m["mix_margin"][s], float)


def test_free_gen_seeds_paired_across_kinds(tiny_model, char_tok, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ev, "generate_counted", _fake_gen(["the answer is C"], [8], seen))
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(), free_gen_tokens=16)
    # order: unsteered, then per kind (trained, source_full, trained_random) c = +1, c = -1
    plus = [seen[i]["rand"] for i in (1, 3, 5)]
    minus = [seen[i]["rand"] for i in (2, 4, 6)]
    assert len(set(plus)) == 1 and len(set(minus)) == 1  # same seed per c for every kind
    assert plus[0] != minus[0] and seen[0]["rand"] not in (plus[0], minus[0])


def test_eval_json_written_before_free_generation(tiny_model, char_tok, tmp_path, monkeypatch):
    path = tmp_path / "out" / "trained" / "run" / "eval.json"
    snapshots = []
    real = _fake_gen(["the answer is C"], [8], [])

    def fake(*a, **kw):
        if not snapshots:
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
        return real(*a, **kw)

    monkeypatch.setattr(ev, "generate_counted", fake)
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(), free_gen_tokens=16)
    (early,) = snapshots
    assert early["free_gen"]["status"] == "pending"
    t = early["traits"]["t1"]
    assert "free_gen" not in t and "length@c=+1" in t["pass"]["skipped"]
    assert t["kinds"]["trained"]["c=+1"]["margins"] == res["traits"]["t1"]["kinds"]["trained"]["c=+1"]["margins"]
    assert early["summary"] == {"t1": t["pass"]["passed"]}

    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["free_gen"]["status"] == "done" and final["traits"]["t1"]["pass"]["skipped"] == []
    assert "free_gen" in final["traits"]["t1"]
    assert final["timing"]["teacher_forced_s"] >= 0 and final["timing"]["free_gen_s"] >= 0


def test_eval_without_free_gen_status_disabled(tiny_model, char_tok, tmp_path):
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": td.source}, _refs(2), free_gen=False)
    assert res["free_gen"]["status"] == "disabled" and res["timing"]["free_gen_s"] is None


def test_eval_margins_follow_logprob_norm(tiny_model, char_tok, tmp_path):
    td = _trait("t1", 1)  # held answers "a" * 6 / "b" * 6: 6 tokens each
    v = {"t1": 2.0 * td.source}
    s = ev.evaluate(tiny_model, char_tok, _cfg(tmp_path, run_name="s"), {"t1": td}, v, _refs(2), free_gen=False)
    m = ev.evaluate(tiny_model, char_tok, _cfg(tmp_path, run_name="m", logprob_norm="mean"), {"t1": td}, v,
                    _refs(2), free_gen=False)
    for kind in ("trained", "source_full"):
        for c in CS:
            want = [x / 6 for x in s["traits"]["t1"]["kinds"][kind][c]["margins"]]
            assert m["traits"]["t1"]["kinds"][kind][c]["margins"] == pytest.approx(want, rel=1e-4, abs=1e-6)
    # KL is per token either way
    assert m["traits"]["t1"]["kinds"]["trained"]["c=+1"]["kls"] == s["traits"]["t1"]["kinds"]["trained"]["c=+1"]["kls"]
