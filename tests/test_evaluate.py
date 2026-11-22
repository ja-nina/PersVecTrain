from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from persvecfast.extract import RANDOM_CONTROL_SEED, random_control

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


def _trait_eval(trained_m, source_m, trained_kl, source_kl, random_kl, tokens=None, correct=None) -> dict:
    """tokens: {"unsteered": u, "trained": t, "source_full": s, "trained_random": r}, each a per-prompt token
    list (a number: that many tokens on every prompt), or None for no free generation. correct: the same for
    the 0 / 1 correctness lists (default: every answer correct)."""
    kinds = {"trained": _kind(trained_m, trained_kl), "source_full": _kind(source_m, source_kl),
             "trained_random": _kind([0.0] * len(trained_m), random_kl)}
    te = {"kinds": kinds}
    if tokens is not None:
        n = len(trained_m)
        tok = {k: (list(v) if isinstance(v, (list, tuple)) else [v] * n) for k, v in tokens.items()}
        cor = {k: list((correct or {}).get(k, [1] * n)) for k in tok}
        fg = {"unsteered": {"tokens": tok["unsteered"], "correct": cor["unsteered"]}}
        for k in ("trained", "source_full", "trained_random"):
            fg[k] = {c: {"tokens": tok[k], "correct": cor[k]} for c in ("c=+1", "c=-1")}
        te["free_gen"] = fg
    return te


GOOD_TOKENS = {"unsteered": 100.0, "trained": 105.0, "source_full": 150.0, "trained_random": 110.0}


def test_pass_rule_all_better_passes():
    te = _trait_eval([2, 3, 4, 5], [1, 2, 3, 4], [0.1, 0.2, 0.1, 0.2], [1.0, 1.1, 0.9, 1.0],
                     [0.5, 0.6, 0.5, 0.6], GOOD_TOKENS)
    r = ev.pass_rule(te)
    assert r["passed"] is True and r["failed"] == [] and r["skipped"] == []
    assert set(r["checks"]) == {"c=+1", "c=-1"}
    assert set(r["checks"]["c=+1"]) == {"persona", "kl", "length", "accuracy", "random"}
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
    assert r["skipped"] == ["length@c=+1", "accuracy@c=+1", "random-length@c=+1",
                            "length@c=-1", "accuracy@c=-1", "random-length@c=-1"]
    assert "length" not in r["checks"]["c=+1"] and "accuracy" not in r["checks"]["c=+1"]


def _acc_eval(trained_correct, source_correct) -> dict:
    n = len(trained_correct)
    return _trait_eval([2] * n, [1] * n, [0.1] * n, [1.0] * n, [0.5] * n, GOOD_TOKENS,
                       correct={"trained": trained_correct, "source_full": source_correct})


def test_pass_rule_accuracy_tie_within_one_se_passes():
    # correct_trained - correct_source: 0, -1, 0, -1, 0, +1: mean -1/6, SE about 0.31
    r = ev.pass_rule(_acc_eval([1, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0]))
    acc = r["checks"]["c=+1"]["accuracy"]
    assert acc["diff_mean"] == pytest.approx(-1 / 6) and -acc["diff_se"] <= acc["diff_mean"] < 0
    assert acc["passed"] is True and r["passed"] is True


def test_pass_rule_accuracy_lower_beyond_one_se_fails():
    # 0, -1, -1, -1, -1, 0: mean -2/3, SE about 0.21
    r = ev.pass_rule(_acc_eval([1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]))
    acc = r["checks"]["c=+1"]["accuracy"]
    assert acc["diff_mean"] == pytest.approx(-2 / 3) and acc["diff_mean"] < -acc["diff_se"]
    assert r["failed"] == ["accuracy@c=+1", "accuracy@c=-1"] and r["passed"] is False


def test_pass_rule_free_gen_needs_the_same_questions():
    te = _acc_eval([1, 1], [1, 1])
    for k in ("trained", "source_full", "trained_random"):
        for c in ("c=+1", "c=-1"):
            te["free_gen"][k][c]["qid"] = ["a", "b"]
    te["free_gen"]["unsteered"]["qid"] = ["a", "b"]
    assert ev.pass_rule(te)["passed"] is True
    te["free_gen"]["source_full"]["c=+1"]["qid"] = ["b", "a"]
    with pytest.raises(ValueError):
        ev.pass_rule(te)


def test_mean_se():
    assert ev.mean_se([1.0, 3.0]) == (2.0, pytest.approx(1.0))
    assert ev.mean_se([5.0]) == (5.0, None)
    assert ev.mean_se([]) == (None, None)


# ---------------------------------------------------------------- generate (reference)


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


def test_evaluate_tiny_every_kind_and_c(tiny_model, char_tok, tmp_path, capsys):
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    trained = {"t1": 2.0 * td.source + 0.1}
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, trained, _refs())

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

    # no free generation in evaluate: pending until `freegen` runs, its checks skipped
    assert res["free_gen"] == {"enabled": True, "status": "pending"} and "free_gen" not in t
    assert "accuracy@c=+1" in t["pass"]["skipped"] and "length@c=-1" in t["pass"]["skipped"]
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


def test_evaluate_keeps_previous_free_gen_block(tiny_model, char_tok, tmp_path):
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    first = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(2))
    blk = {"n": 2, "mean_tokens": 8.0, "cut_share": 0.0, "accuracy": 1.0, "accuracy_se": 0.0,
           "qid": ["r0", "r1"], "tokens": [8, 8], "cut": [False, False], "answer": ["C", "C"], "correct": [1, 1]}
    previous = json.loads(json.dumps(first))
    previous["free_gen"] = {"enabled": True, "status": "done", "max_tokens": 15360, "random_dirs": 5,
                            "unsteered": blk}
    previous["timing"]["free_gen_s"] = 12.5
    previous["traits"]["t1"]["free_gen"] = {"unsteered": blk, **{k: {"c=+1": blk, "c=-1": blk}
                                                                 for k in ("trained", "source_full", "trained_random")}}
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(2), previous=previous)
    assert res["free_gen"] == previous["free_gen"] and res["timing"]["free_gen_s"] == 12.5
    t = res["traits"]["t1"]
    assert t["free_gen"] == previous["traits"]["t1"]["free_gen"]
    assert t["pass"]["skipped"] == [] and "accuracy" in t["pass"]["checks"]["c=+1"]
    assert t["kinds"]["trained"]["c=+1"]["margins"] == first["traits"]["t1"]["kinds"]["trained"]["c=+1"]["margins"]
    # free_gen False: the previous block is dropped
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(2), free_gen=False,
                      previous=previous)
    assert res["free_gen"]["status"] == "disabled" and "free_gen" not in res["traits"]["t1"]


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


def test_evaluate_rerun_after_disabled_resets_to_pending(tiny_model, char_tok, tmp_path):
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    off = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(2), free_gen=False)
    assert off["free_gen"]["status"] == "disabled"
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(2), previous=off)
    assert res["free_gen"] == {"enabled": True, "status": "pending"} and "free_gen" not in res["traits"]["t1"]


# ---------------------------------------------------------------- 5 random directions


def test_random_seeds_and_set():
    assert ev.random_seeds(LAYER) == [RANDOM_CONTROL_SEED + LAYER + 1000 * k for k in range(5)]
    v = 3.0 * torch.randn(16, generator=torch.Generator().manual_seed(1))
    dirs = ev.random_set(v, LAYER)
    assert len(dirs) == 5
    for d in dirs:
        assert float(d.norm()) == pytest.approx(float(v.norm()), rel=1e-5)
    assert torch.equal(dirs[0], random_control(v, RANDOM_CONTROL_SEED + LAYER))  # k = 0 is today's twin
    assert len({tuple(d.tolist()) for d in dirs}) == 5


def test_random_kinds_are_the_mean_of_five_directions(tiny_model, char_tok, tmp_path):
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    v = 1.5 * td.source + 0.1
    refs = _refs(2)
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": v}, refs, free_gen=False)
    assert res["random_dirs"] == 5
    kinds = res["traits"]["t1"]["kinds"]
    sources = {"trained_random": v, "source_random": td.source}
    held = [(encode(char_tok, p.question, p.pos, MAX_TOK) + encode(char_tok, p.question, p.neg, MAX_TOK))
            for p in td.held]
    with torch.no_grad():
        for kind, base in sources.items():
            dirs = [d.float() for d in ev.random_set(base, LAYER)]
            for c in (1.0, -1.0):
                key = ev.c_key(c)
                per_dir_kl = []
                for d in dirs:
                    kl = []
                    for r in refs:
                        ids, start = encode(char_tok, r.question, r.answer_text, MAX_TOK)
                        lp0 = ev.answer_logprobs(tiny_model, ids, start)
                        with steer(tiny_model, LAYER, c * d):
                            lpc = ev.answer_logprobs(tiny_model, ids, start)
                        kl.append(float(ev.kl_loss(lp0, lpc)))
                    per_dir_kl.append(sum(kl) / len(kl))
                row = kinds[kind][key]
                assert row["per_direction"] == pytest.approx(per_dir_kl, rel=1e-4, abs=1e-7)
                assert len(row["per_direction"]) == 5 and len(row["per_direction_margin"]) == 5
                assert row["kl_mean"] == pytest.approx(sum(per_dir_kl) / 5, rel=1e-4, abs=1e-7)
                assert sum(row["per_direction_margin"]) / 5 == pytest.approx(row["margin_mean"], rel=1e-4, abs=1e-6)
                assert len(set(round(x, 9) for x in per_dir_kl)) > 1  # the directions differ
            # c = +-0.5: k = 0 only, no per-direction block
            for key in ("c=-0.5", "c=+0.5"):
                assert "per_direction" not in kinds[kind][key]
            # per pair margin at c = +1 by hand, mean of the 5 directions
            ms = []
            for d in dirs:
                lp0 = ev._pair_logprobs(tiny_model, cfg, None, held[0])
                ms.append(ev._margin(tiny_model, cfg, d, 1, held[0], lp0))
            assert kinds[kind]["c=+1"]["margins"][0] == pytest.approx(sum(ms) / 5, rel=1e-4, abs=1e-6)
    # the other kinds carry no per-direction block, and k = 0 stays the reported twin
    assert "per_direction" not in kinds["trained"]["c=+1"]
    assert res["traits"]["t1"]["norms"]["trained_random"] == pytest.approx(float(v.norm()))


def test_evaluate_old_single_twin_free_gen_goes_back_to_pending(tiny_model, char_tok, tmp_path):
    cfg = _cfg(tmp_path)
    td = _trait("t1", 1)
    first = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(2))
    blk = {"n": 2, "mean_tokens": 8.0, "cut_share": 0.0, "accuracy": 1.0, "accuracy_se": 0.0,
           "qid": ["r0", "r1"], "tokens": [8, 8], "cut": [False, False], "answer": ["C", "C"], "correct": [1, 1]}
    previous = json.loads(json.dumps(first))
    previous["free_gen"] = {"enabled": True, "status": "done", "max_tokens": 15360, "unsteered": blk}  # no random_dirs
    previous["traits"]["t1"]["free_gen"] = {"unsteered": blk, **{k: {"c=+1": blk, "c=-1": blk}
                                                                 for k in ("trained", "source_full", "trained_random")}}
    res = ev.evaluate(tiny_model, char_tok, cfg, {"t1": td}, {"t1": 2.0 * td.source}, _refs(2), previous=previous)
    assert res["free_gen"]["status"] == "pending" and res["free_gen"]["max_tokens"] == 15360
    assert res["traits"]["t1"]["free_gen"] == previous["traits"]["t1"]["free_gen"]
