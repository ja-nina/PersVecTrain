from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from persvecfast.engine import Result
from persvecfast.extract import RANDOM_CONTROL_SEED, random_control
from persvecfast.models import MODELS, request_seed

from persvectrain import cli
from persvectrain import freegen as fgm
from persvectrain.data import supergpqa_prompts
from persvectrain.output import config_dict, prepare_run, write_vectors
from persvectrain.train import TrainConfig
from tests.conftest import CharTokenizer
from tests.test_cli import MODEL, _fake_trait

LAYER = 1
HELD = ["s4", "s0", "s3"]  # train.json's order (not sorted); s0 has 10 options, the others 4
KINDS = ("trained", "source_full", "trained_random")


# ---------------------------------------------------------------- parse_answer


@pytest.mark.parametrize("text,n,letter", [
    ("Reasoning... ANSWER: C", 4, "C"),
    ("so the final answer: c", 4, "C"),
    ("ANSWER: (D)", 4, "D"),
    ("**ANSWER:** B", 4, "B"),
    ("ANSWER: A at first. Re-checking. ANSWER: B", 4, "B"),
    ("ANSWER: E", 4, None),
    ("ANSWER: J", 10, "J"),
    ("The option is C", 4, None),
    ("ANSWER: Because", 4, None),  # a letter followed by another letter is a word
    ("", 4, None),
])
def test_parse_answer(text, n, letter):
    assert fgm.parse_answer(text, n) == letter


def test_instruction_lists_the_question_letters():
    assert fgm.instruction(4) == "\n\nEnd your reply with 'ANSWER: <letter>', one of A, B, C, D."


# ---------------------------------------------------------------- a finished run, by hand


def _pool(path: Path, n: int = 6) -> Path:
    rows = [{"id": f"s{i}", "question": f"Question {i}?", "options": [f"opt{j}" for j in range(10 if i == 0 else 4)],
             "answer": "C"} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _kinds() -> dict:
    def kind(m, kl):
        return {c: {"margins": [m] * 3, "kls": [kl] * 3, "margin_mean": m} for c in ("c=+1", "c=-1")}
    return {"trained": kind(2.0, 0.1), "source_full": kind(1.0, 1.0), "trained_random": kind(0.0, 0.5)}


@pytest.fixture
def run(tmp_path):
    out = tmp_path / "out"
    src = out / "persona" / MODEL / "bf16"
    traits = ["assertive", "calm"]
    for t in traits:
        _fake_trait(src, t)
    pool = _pool(tmp_path / "pool.jsonl")
    ref = out / "trained" / "reference" / f"{MODEL}_pool260.jsonl"
    ref.parent.mkdir(parents=True)
    ref.write_text("".join(json.dumps({"id": it["id"], "question": it["question"], "answer_text": "x",
                                       "gold": it["answer"]}) + "\n" for it in supergpqa_prompts(pool)),
                   encoding="utf-8")
    cfg = TrainConfig(model=MODEL, layer=LAYER, mode="solo", traits=traits, source=out.resolve(), out=out.resolve(),
                      run_name="run1", seed=0)
    vec_dir = prepare_run(cfg)
    g = torch.Generator().manual_seed(5)
    for t in traits:
        write_vectors(vec_dir, t, LAYER, 3 * torch.randn(16, generator=g), src, "run1")
    root = out / "trained" / "run1"
    meta = {"config": config_dict(cfg), "seed": 0, "finished": True, "dropped_traits": {"evil": "3 kept pairs < 20"},
            "sources": {"reference": str(ref)},
            "splits": {"reference": {"train": ["s1", "s2", "s5"], "held": HELD}}}
    (root / "train.json").write_text(json.dumps(meta), encoding="utf-8")
    ev = {"run_name": "run1", "free_gen": {"enabled": True, "status": "pending"},
          "timing": {"teacher_forced_s": 1.0, "free_gen_s": None},
          "traits": {t: {"kinds": _kinds(), "pass": {"passed": False, "failed": [], "skipped": ["length@c=+1"]}}
                     for t in traits}}
    (root / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    return {"out": out, "root": root, "pool": pool, "cfg": cfg, "src": src, "vec_dir": vec_dir}


class FakeEngine:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, requests):
        self.calls.append(list(requests))
        return [Result(r.key, *self.reply(r)) for r in requests]


def _fake(monkeypatch, reply=lambda r: ("Thinking. ANSWER: C", 10, "stop")):
    engine = FakeEngine(reply)
    opened = []

    @contextmanager
    def open_engine(spec, kind):
        opened.append((spec.key, kind))
        yield engine

    monkeypatch.setattr(fgm, "load_tokenizer", lambda key: CharTokenizer())
    monkeypatch.setattr(fgm, "default_open_engine", lambda: open_engine)
    return engine, opened


def _freegen(run, **kw):
    return fgm.run_freegen(run["out"], "run1", pool=run["pool"], **kw)


def _reqs(engine):
    return [r for call in engine.calls for r in call]


# ---------------------------------------------------------------- requests


def test_requests_keys_seeds_and_steering(run, monkeypatch):
    engine, opened = _fake(monkeypatch)
    _freegen(run, max_tokens=64)
    reqs = _reqs(engine)
    assert opened == [(MODEL, "steer")]
    assert len(reqs) == len(HELD) * (1 + 6 * 2)  # 100 x (1 + 6 x n_traits) at full size
    keys = [r.key for r in reqs]
    assert len(set(keys)) == len(keys)
    want = {("-", "unsteered", 0.0, q) for q in HELD} | {(s, k, c, q) for s in ("assertive", "calm") for k in KINDS
                                                        for c in (1.0, -1.0) for q in HELD}
    assert set(keys) == want
    assert all(r.max_tokens == 64 for r in reqs)
    by_key = {r.key: r for r in reqs}
    for q in HELD:
        u = by_key[("-", "unsteered", 0.0, q)]
        assert u.steer is None and u.seed == request_seed(0, "freegen", "unsteered", q)
        for c in (1.0, -1.0):
            seeds = {by_key[(s, k, c, q)].seed for s in ("assertive", "calm") for k in KINDS}
            assert seeds == {request_seed(0, "freegen", c, q)}  # paired: one seed per (c, question)
        assert request_seed(0, "freegen", 1.0, q) != request_seed(0, "freegen", -1.0, q)
    r = by_key[("calm", "source_full", -1.0, "s3")]
    assert r.steer == {"layer": LAYER, "coeff": -1.0,
                       "vector_path": str((run["root"] / "freegen_vectors" / "calm_source_full.pt").resolve())}
    # prompt: the held question + the games' instruction with that question's letters, neutral chat template
    p0, p4 = by_key[("-", "unsteered", 0.0, "s0")].prompt, by_key[("-", "unsteered", 0.0, "s4")].prompt
    assert p0.startswith("<s>") and "Question 0?" in p0 and p0.endswith("one of A, B, C, D, E, F, G, H, I, J.<a>")
    assert p4.endswith("Which option is correct?\n\nEnd your reply with 'ANSWER: <letter>', one of A, B, C, D.<a>")


def test_requests_go_in_chunks_of_batch_size(run, monkeypatch):
    engine, opened = _fake(monkeypatch)
    monkeypatch.setitem(fgm.MODELS, MODEL, dataclasses.replace(MODELS[MODEL], batch_size=5))
    _freegen(run)
    sizes = [len(c) for c in engine.calls]
    assert opened == [(MODEL, "steer")]  # one engine for every chunk
    assert sum(sizes) == 39 and max(sizes) == 5 and len(sizes) == 8
    assert all(r.max_tokens == 15360 for r in _reqs(engine))


def test_vector_files_hold_the_right_tensors(run, monkeypatch):
    _fake(monkeypatch)
    _freegen(run)
    d = run["root"] / "freegen_vectors"
    for s in ("assertive", "calm"):
        trained = torch.load(run["vec_dir"] / "vllm_hook" / f"{s}_L{LAYER}.pt")["dir"]
        assert torch.equal(torch.load(d / f"{s}_trained.pt")["dir"], trained)
        assert torch.equal(torch.load(d / f"{s}_source_full.pt")["dir"], torch.load(run["src"] / f"{s}.pt")[LAYER])
        twin = torch.load(d / f"{s}_trained_random.pt")["dir"]
        assert torch.equal(twin, random_control(trained, RANDOM_CONTROL_SEED + LAYER))
        assert torch.equal(twin, torch.load(run["vec_dir"] / "vllm_hook" / f"{s}_L{LAYER}_random.pt")["dir"])


# ---------------------------------------------------------------- results in eval.json


def _reply(r):
    """unsteered right and short; trained right; source_full wrong at c=+1 on s0 and cut; random: no marker."""
    slug, kind, c, q = r.key
    if kind == "unsteered":
        return "ANSWER: C", 100, "stop"
    if kind == "trained":
        return "**ANSWER:** c", 110, "stop"
    if kind == "source_full":
        return ("ANSWER: A", 400, "length") if (c > 0 and q == "s0") else ("ANSWER: (C)", 300, "stop")
    return "no idea", 120, "stop"


def test_pending_to_done_with_per_item_lists(run, monkeypatch, capsys):
    _fake(monkeypatch, _reply)
    train_json = (run["root"] / "train.json").read_bytes()
    vec_bytes = {p: p.read_bytes() for p in run["vec_dir"].rglob("*") if p.is_file()}
    res = _freegen(run, max_tokens=500)
    ev = json.loads((run["root"] / "eval.json").read_text(encoding="utf-8"))
    assert ev == res
    fg = ev["free_gen"]
    assert fg["status"] == "done" and fg["max_tokens"] == 500 and fg["n_held"] == 3
    assert fg["sampling"]["presence_penalty"] == MODELS[MODEL].presence_penalty and fg["presence_penalty_applied"]
    assert fg["unsteered"]["qid"] == HELD and fg["unsteered"]["accuracy"] == 1.0
    t = ev["traits"]["assertive"]["free_gen"]
    assert t["unsteered"] == fg["unsteered"]
    sf = t["source_full"]["c=+1"]
    assert sf["qid"] == HELD and sf["tokens"] == [300, 400, 300] and sf["cut"] == [False, True, False]
    assert sf["answer"] == ["C", "A", "C"] and sf["correct"] == [True, False, True]
    assert sf["accuracy"] == pytest.approx(2 / 3) and sf["cut_share"] == pytest.approx(1 / 3) and sf["n"] == 3
    assert t["source_full"]["c=-1"]["accuracy"] == 1.0
    assert t["trained"]["c=+1"]["answer"] == ["C"] * 3 and t["trained"]["c=+1"]["mean_tokens"] == 110.0
    assert t["trained_random"]["c=-1"]["answer"] == [None] * 3 and t["trained_random"]["c=-1"]["accuracy"] == 0.0
    # pass rule recomputed with the free-gen checks
    p = ev["traits"]["assertive"]["pass"]
    assert p["skipped"] == [] and p["checks"]["c=+1"]["accuracy"]["passed"] is True
    assert p["checks"]["c=+1"]["length"]["passed"] is True  # |110-100| < |300..400-100|
    assert p["passed"] is True and ev["summary"] == {"assertive": True, "calm": True}
    assert ev["traits"]["assertive"]["kinds"] == _kinds() and ev["timing"]["teacher_forced_s"] == 1.0
    assert ev["timing"]["free_gen_s"] >= 0
    # train.json and the run's vectors untouched
    assert (run["root"] / "train.json").read_bytes() == train_json
    assert {p: p.read_bytes() for p in run["vec_dir"].rglob("*") if p.is_file()} == vec_bytes
    text = capsys.readouterr().out
    assert "evil" in text and "dropped" in text  # train's dropped trait: a message, no crash
    assert "unsteered" in text and "trained_random" in text and "pass assertive: PASS" in text


def test_rerun_overwrites_only_the_free_gen_block(run, monkeypatch):
    _fake(monkeypatch)
    _freegen(run)
    first = json.loads((run["root"] / "eval.json").read_text(encoding="utf-8"))
    assert first["traits"]["calm"]["free_gen"]["trained"]["c=+1"]["tokens"] == [10] * 3
    train_json = (run["root"] / "train.json").read_bytes()
    vec_bytes = {p: p.read_bytes() for p in run["vec_dir"].rglob("*") if p.is_file()}

    engine, _ = _fake(monkeypatch, _reply)
    _freegen(run)
    again = json.loads((run["root"] / "eval.json").read_text(encoding="utf-8"))
    assert again["free_gen"]["status"] == "done" and len(_reqs(engine)) == 39
    assert again["traits"]["calm"]["free_gen"]["trained"]["c=+1"]["tokens"] == [110] * 3
    for s in ("assertive", "calm"):
        assert again["traits"][s]["kinds"] == first["traits"][s]["kinds"]
    assert {k: v for k, v in again.items() if k not in ("free_gen", "traits", "summary", "timing")} == \
        {k: v for k, v in first.items() if k not in ("free_gen", "traits", "summary", "timing")}
    assert again["timing"]["teacher_forced_s"] == first["timing"]["teacher_forced_s"]
    assert (run["root"] / "train.json").read_bytes() == train_json
    assert {p: p.read_bytes() for p in run["vec_dir"].rglob("*") if p.is_file()} == vec_bytes


# ---------------------------------------------------------------- skipped runs


def _assert_untouched(run, before):
    assert {p: p.read_bytes() for p in run["root"].rglob("*") if p.is_file()} == before
    assert not (run["root"] / "freegen_vectors").exists()


def test_disabled_run_is_skipped_with_a_message(run, monkeypatch, capsys):
    engine, opened = _fake(monkeypatch)
    path = run["root"] / "eval.json"
    ev = json.loads(path.read_text(encoding="utf-8"))
    ev["free_gen"] = {"enabled": False, "status": "disabled"}
    path.write_text(json.dumps(ev), encoding="utf-8")
    before = {p: p.read_bytes() for p in run["root"].rglob("*") if p.is_file()}
    assert _freegen(run) is None
    assert "--no-free-gen" in capsys.readouterr().out and opened == []
    _assert_untouched(run, before)
    args = ["freegen", "--out", str(run["out"]), "--run-name", "run1", "--pool", str(run["pool"])]
    assert cli.main(args) == 1
    _assert_untouched(run, before)


@pytest.mark.parametrize("missing", ["eval.json", "train.json"])
def test_run_without_eval_or_train_json_is_skipped(run, monkeypatch, capsys, missing):
    _, opened = _fake(monkeypatch)
    (run["root"] / missing).unlink()
    before = {p: p.read_bytes() for p in run["root"].rglob("*") if p.is_file()}
    assert cli.main(["freegen", "--out", str(run["out"]), "--run-name", "run1", "--pool", str(run["pool"])]) == 1
    assert missing in capsys.readouterr().out and opened == []
    _assert_untouched(run, before)


def test_unfinished_run_is_skipped(run, monkeypatch, capsys):
    _, opened = _fake(monkeypatch)
    path = run["root"] / "train.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["finished"] = False
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert _freegen(run) is None and opened == [] and "did not finish" in capsys.readouterr().out


def test_old_reference_is_refused(run, monkeypatch):
    _fake(monkeypatch)
    path = run["root"] / "train.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    old = Path(meta["sources"]["reference"]).with_name(f"{MODEL}.jsonl")
    old.write_text(Path(meta["sources"]["reference"]).read_text(encoding="utf-8"), encoding="utf-8")
    meta["sources"]["reference"] = str(old)
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        _freegen(run)


def test_reference_differing_from_the_pool_is_refused(run, monkeypatch):
    _fake(monkeypatch)
    rows = [json.loads(l) for l in run["pool"].read_text(encoding="utf-8").splitlines()]
    rows[3]["question"] = "Another question?"
    run["pool"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        _freegen(run)


def test_engine_missing_answers_raises(run, monkeypatch):
    engine, _ = _fake(monkeypatch)
    real = engine.generate
    engine.generate = lambda reqs: real(reqs)[:-1]
    before = (run["root"] / "eval.json").read_bytes()
    with pytest.raises(RuntimeError):
        _freegen(run)
    assert (run["root"] / "eval.json").read_bytes() == before
