from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
import torch
from persvecfast.judge import JudgeError, Judges
from persvecfast.models import request_seed

from persvectrain import cli
from persvectrain import persona as pm
from persvectrain.evaluate import c_key
from tests.conftest import CharTokenizer
from tests.test_freegen import FakeEngine, MODEL, run  # noqa: F401  (run is a fixture)

HELD_PAIRS = {"assertive": [0, 1, 2], "calm": [4, 5, 6]}
QIDS = [f"{s}:{i}" for s, idx in HELD_PAIRS.items() for i in idx]
N_Q = len(QIDS)
ELICIT = {"assertive": "BE ASSERTIVE", "calm": "BE CALM"}


class FakeClient:
    async def aclose(self):
        pass


class FakeJudgeWorld:
    """Replaces the judge calls: score = number of 'x' in the answer text, coherence = 80 unless fail / none say so."""

    def __init__(self):
        self.calls = []
        self.fail_after = None  # raise JudgeError once this many expression calls were made
        self.none_for = set()

    async def expression(self, judges, question, response, trait, description):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise JudgeError("402 no credit")
        self.calls.append((question, response, trait))
        return None if response in self.none_for else response.count("x")

    async def coherence(self, judges, question, response):
        return 80


@pytest.fixture
def prun(run, monkeypatch):  # noqa: F811
    """The freegen test run with per-trait held pairs in train.json and a pending persona_judge in eval.json."""
    root = run["root"]
    meta = json.loads((root / "train.json").read_text(encoding="utf-8"))
    meta["config"]["traits"] = ["assertive", "calm"]
    for s, idx in HELD_PAIRS.items():
        meta["splits"][s] = {"train": [9, 10], "held": idx}
    (root / "train.json").write_text(json.dumps(meta), encoding="utf-8")
    ev = json.loads((root / "eval.json").read_text(encoding="utf-8"))
    ev["persona_judge"] = {"status": "pending"}
    (root / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    monkeypatch.setattr(pm, "trait_info", lambda slug: {"trait": slug.title(), "description": f"d {slug}",
                                                        "instruction": {"positive": ELICIT[slug]}})
    return run


def _setup(monkeypatch, reply=lambda r: ("answer xxx", 7, "stop"), make_judges=True):
    engine = FakeEngine(reply)
    opened = []

    @contextmanager
    def open_engine(spec, kind):
        opened.append(kind)
        yield engine

    world = FakeJudgeWorld()
    monkeypatch.setattr(pm.fg, "load_tokenizer", lambda key: CharTokenizer())
    monkeypatch.setattr(pm, "score_expression", world.expression)
    monkeypatch.setattr(pm, "score_coherence", world.coherence)
    judges = lambda conc: Judges(make_client=FakeClient)  # noqa: E731
    return engine, opened, world, open_engine, judges


def _persona(run, open_engine, judges, **kw):
    return pm.run_persona(run["out"], "run1", open_engine=open_engine, make_judges=judges, **kw)


def _reqs(engine):
    return [r for call in engine.calls for r in call]


def test_questions_union_in_trait_then_pair_order(prun):
    meta = json.loads((prun["root"] / "train.json").read_text(encoding="utf-8"))
    qs = pm.questions(meta, prun["src"])
    assert [q for q, _ in qs] == QIDS
    assert dict(qs)["calm:5"] == "q5?"


def test_requests_count_seeds_prompts_and_rotation(prun, monkeypatch):
    engine, opened, world, oe, judges = _setup(monkeypatch)
    _persona(prun, oe, judges, max_tokens=64)
    reqs = _reqs(engine)
    assert opened == ["steer"] and len(reqs) == N_Q * 2 * (2 + 3 * 2)
    assert all(r.max_tokens == 64 for r in reqs)
    by = {r.key: r for r in reqs}
    assert len(by) == len(reqs)
    for qi, qid in enumerate(QIDS):
        # seeds paired across kinds and traits at one c / question; unsteered / elicited have their own
        for c in (1.0, -1.0):
            assert {by[(s, k, c, qid)].seed for s in HELD_PAIRS for k in pm.FREE_GEN_KINDS} == \
                {request_seed(0, "persona", c, qi)}
        assert by[("calm", "unsteered", 0.0, qid)].seed == request_seed(0, "persona", "neutral", qi)
        assert by[("calm", "elicited", 0.0, qid)].seed == request_seed(0, "persona", "elicited", qi)
        assert by[("calm", "unsteered", 0.0, qid)].steer is None and by[("calm", "elicited", 0.0, qid)].steer is None
    r = by[("calm", "trained", 1.0, "calm:5")]
    assert r.prompt == "<s>You are a helpful assistant.<u>q5?<a>"  # neutral system prompt at +1, no answer instruction
    assert by[("calm", "trained", -1.0, "calm:5")].prompt == "<s>BE CALM<u>q5?<a>"  # eliciting at -1
    assert by[("assertive", "elicited", 0.0, "assertive:0")].prompt == "<s>BE ASSERTIVE<u>q0?<a>"
    assert by[("assertive", "unsteered", 0.0, "assertive:0")].prompt.startswith("<s>You are a helpful assistant.")
    # rotation by question position: k = i % 5
    d = prun["root"] / "freegen_vectors"
    for qi, qid in enumerate(QIDS):
        assert by[("calm", "trained_random", 1.0, qid)].steer["vector_path"] == \
            str((d / f"calm_trained_random_k{qi % 5}.pt").resolve())
        assert by[("calm", "source_full", -1.0, qid)].steer["vector_path"] == \
            str((d / "calm_source_full.pt").resolve())
    assert by[("calm", "trained", -1.0, "calm:4")].steer["coeff"] == -1.0


def test_judge_scores_land_in_eval_json_and_pass_rule(prun, monkeypatch):
    # trained answers carry more x than source_full; random none; judge@+1 passes, judge@-1 (suppression) fails
    def reply(r):
        slug, kind, c, qid = r.key
        n = {"trained": 6, "source_full": 3}.get(kind, 0 if kind != "elicited" else 9)
        return ("x" * n + "a", 5, "stop")

    engine, _, world, oe, judges = _setup(monkeypatch, reply)
    ev = _persona(prun, oe, judges)
    assert ev["persona_judge"]["status"] == "done" and ev["persona_judge"]["n_questions"] == N_Q
    assert ev["persona_judge"]["max_tokens"] == 1024 and ev["persona_judge"]["judge_model"] == pm.JUDGE_MODEL
    t = ev["traits"]["calm"]["persona_judge"]
    assert t["elicited"]["score_mean"] == 9 and t["unsteered"]["score_mean"] == 0
    assert t["trained"]["c=+1"]["score"] == [6] * N_Q and t["trained"]["c=+1"]["qid"] == QIDS
    assert t["trained"]["c=-1"]["coherence_mean"] == 80 and t["trained"]["c=+1"]["n_tokens"] == [5] * N_Q
    assert t["trained"]["c=+1"]["score_se"] == 0.0
    p = ev["traits"]["calm"]["pass"]
    assert p["checks"]["c=+1"]["judge"]["passed"] is True
    assert p["checks"]["c=-1"]["judge"]["passed"] is False and "judge@c=-1" in p["failed"]
    assert p["checks"]["c=+1"]["coherence"]["passed"] is True
    assert not any(s.startswith(("judge", "coherence")) for s in p["skipped"])
    on_disk = json.loads((prun["root"] / "eval.json").read_text(encoding="utf-8"))
    assert on_disk["persona_judge"]["status"] == "done"
    assert (prun["root"] / pm.ANSWERS_FILE).exists() and (prun["root"] / pm.JUDGED_FILE).exists()


def test_empty_answer_scores_zero_without_a_judge_call(prun, monkeypatch):
    engine, _, world, oe, judges = _setup(monkeypatch, lambda r: ("", 0, "stop"))
    ev = _persona(prun, oe, judges)
    assert world.calls == []
    assert ev["traits"]["calm"]["persona_judge"]["trained"]["c=+1"]["score"] == [0] * N_Q


def test_missing_key_fails_before_the_engine_opens(prun, monkeypatch):
    engine, opened, world, oe, _ = _setup(monkeypatch)

    def no_key(conc):
        raise RuntimeError("OPENROUTER_API_KEY is not set (export it or put it in .env)")

    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        _persona(prun, oe, no_key)
    assert opened == [] and engine.calls == [] and not (prun["root"] / pm.ANSWERS_FILE).exists()


def test_judge_error_keeps_answers_and_rerun_judges_only_the_rest(prun, monkeypatch):
    engine, opened, world, oe, judges = _setup(monkeypatch)
    world.fail_after = 40
    with pytest.raises(SystemExit, match="402 no credit") as e:
        _persona(prun, oe, judges, judge_concurrency=1)
    assert "rerun" in str(e.value)
    n_req = N_Q * 2 * 8
    assert len(pm.fg.read_answers(prun["root"] / pm.ANSWERS_FILE)[0]) == n_req  # all answers kept
    judged = pm.read_judged(prun["root"] / pm.JUDGED_FILE)
    assert 0 < len(judged) < n_req
    ev0 = json.loads((prun["root"] / "eval.json").read_text(encoding="utf-8"))
    assert ev0["persona_judge"] == {"status": "pending"}  # not done
    # rerun: no new answers, only the unjudged ones sent to the judge
    world.fail_after, calls_before, engine_calls = None, len(world.calls), len(engine.calls)
    ev = _persona(prun, oe, judges)
    assert len(engine.calls) == engine_calls
    assert len(world.calls) - calls_before == n_req - len(judged)  # exactly the unjudged answers
    assert ev["persona_judge"]["status"] == "done"


def test_none_scores_keep_status_pending_rc_1_and_rerun_resends_only_them(prun, monkeypatch, capsys):
    def reply(r):
        return ("answer xxx" if r.key[1] == "trained" else "answer x", 7, "stop")

    engine, _, world, oe, judges = _setup(monkeypatch, reply)
    world.none_for = {"answer xxx"}  # the trained answers get no score
    ev = _persona(prun, oe, judges)
    n_none = N_Q * 2 * 2  # trained, c = +-1, both traits
    assert ev["persona_judge"]["status"] == "pending" and ev["persona_judge"]["n_unscored"] == n_none
    t = ev["traits"]["calm"]["persona_judge"]
    assert t["trained"]["c=+1"]["score"] == [None] * N_Q and t["trained"]["c=+1"]["n_scored"] == 0
    assert t["source_full"]["c=+1"]["score"] == [1] * N_Q  # the scores so far stay
    assert ev["traits"]["calm"]["pass"]["checks"]["c=+1"]["judge"]["n_pairs"] == 0
    on_disk = json.loads((prun["root"] / "eval.json").read_text(encoding="utf-8"))
    assert on_disk["persona_judge"]["status"] == "pending"
    # the CLI exits 1 while some scores are missing
    monkeypatch.setattr(cli, "run_persona", lambda out, run_name, **kw: pm.run_persona(
        out, run_name, open_engine=oe, make_judges=judges, **kw))
    args = cli.build_parser().parse_args(["persona", "--out", str(prun["out"]), "--run-name", "run1"])
    assert cli.cmd_persona(args) == 1
    # rerun with a working judge: only the None items are sent, then done and rc 0
    world.none_for = set()
    before = len(world.calls)
    assert cli.cmd_persona(args) == 0
    assert len(world.calls) - before == n_none
    done = json.loads((prun["root"] / "eval.json").read_text(encoding="utf-8"))
    assert done["persona_judge"]["status"] == "done" and done["persona_judge"]["n_unscored"] == 0
    assert done["traits"]["calm"]["persona_judge"]["trained"]["c=+1"]["score"] == [3] * N_Q


def test_resume_reuses_answers_and_fresh_regenerates(prun, monkeypatch):
    engine, _, world, oe, judges = _setup(monkeypatch)
    _persona(prun, oe, judges)
    n, j = len(_reqs(engine)), len(world.calls)
    _persona(prun, oe, judges)
    assert len(_reqs(engine)) == n and len(world.calls) == j  # nothing generated or judged again
    _persona(prun, oe, judges, max_tokens=64)  # another max_tokens: new answers, new judge calls
    assert len(_reqs(engine)) == 2 * n and len(world.calls) == 2 * j
    _persona(prun, oe, judges, max_tokens=64, fresh=True)
    assert len(_reqs(engine)) == 3 * n


def test_disabled_run_is_skipped(prun, monkeypatch, capsys):
    engine, opened, world, oe, judges = _setup(monkeypatch)
    ev = json.loads((prun["root"] / "eval.json").read_text(encoding="utf-8"))
    ev["persona_judge"] = {"status": "disabled"}
    (prun["root"] / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    assert _persona(prun, oe, judges) is None
    assert "--no-judge" in capsys.readouterr().out and opened == []


def test_cli_parses_persona_and_no_judge():
    a = cli.build_parser().parse_args(["persona", "--out", "o", "--run-name", "r", "--judge-concurrency", "8",
                                       "--fresh"])
    assert a.max_tokens == 1024 and a.judge_concurrency == 8 and a.fresh is True
    a = cli.build_parser().parse_args(["eval", "--out", "o", "--run-name", "r", "--no-judge"])
    assert a.no_judge is True


def test_report_pending_only_with_the_key(prun):
    from persvectrain.report import is_pending

    assert is_pending({"persona_judge": {"status": "pending"}}) is True
    assert is_pending({"free_gen": {"status": "done"}, "persona_judge": {"status": "done"}}) is False
    assert is_pending({"free_gen": {"status": "done"}}) is False  # an old run without persona_judge
    assert is_pending({"persona_judge": {"status": "disabled"}}) is False
