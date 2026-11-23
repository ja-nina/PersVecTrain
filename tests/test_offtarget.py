from __future__ import annotations

import json

from persvectrain import cli
from persvectrain import offtarget as ot
from tests.test_freegen import run  # noqa: F401  (fixture)
from tests.test_persona import N_Q, QIDS, _persona, _setup, prun  # noqa: F401  (prun is a fixture)


def _reply(r):
    slug, kind, c, qid = r.key
    return ({"trained": "xx", "source_full": "xxxxx"}.get(kind, "") + "a", 5, "stop")


def _fake_judge(calls):
    async def expression(judges, question, response, trait, description):
        calls.append(trait)
        return response.count("x") * (10 if trait == "Calm" else 1)
    return expression


def test_offtarget_judges_other_traits_and_resumes(prun, monkeypatch):
    engine, _, world, oe, judges = _setup(monkeypatch, _reply)
    _persona(prun, oe, judges)
    calls = []
    monkeypatch.setattr(ot, "score_expression", _fake_judge(calls))
    res = ot.run_offtarget(prun["out"], "run1", make_judges=judges)
    # 2 traits: per trait unsteered + 4 kinds at c=+1, N_Q questions, 1 other trait
    assert len(calls) == 2 * 5 * N_Q and res["n_unscored"] == 0 and res["c"] == ["c=+1"]
    a = res["traits"]["assertive"]["calm"]
    assert set(a) == {"unsteered", "trained@c=+1", "source_full@c=+1", "source_at_trained_norm@c=+1",
                      "trained_random@c=+1"}
    assert a["trained@c=+1"]["score"] == [20] * N_Q and a["source_full@c=+1"]["score_mean"] == 50
    assert a["unsteered"]["qid"] == QIDS
    assert "assertive" not in res["traits"]["assertive"]
    assert res["traits"]["calm"]["assertive"]["trained@c=+1"]["score"] == [2] * N_Q
    assert json.loads((prun["root"] / ot.OUT_FILE).read_text(encoding="utf-8"))["traits"]
    # rerun: nothing left to judge; eval.json untouched by offtarget
    before = (prun["root"] / "eval.json").read_text(encoding="utf-8")
    assert ot.run_offtarget(prun["out"], "run1", make_judges=judges)["n_unscored"] == 0
    assert len(calls) == 2 * 5 * N_Q
    assert (prun["root"] / "eval.json").read_text(encoding="utf-8") == before


def test_offtarget_needs_persona_first(prun, capsys):
    assert ot.run_offtarget(prun["out"], "run1", make_judges=lambda c: None) is None
    assert "run `persona` first" in capsys.readouterr().out


def test_cli_parses_offtarget():
    args = cli.build_parser().parse_args(["offtarget", "--out", "o", "--run-names", "a", "b", "--c", "1", "-1"])
    assert args.run_names == ["a", "b"] and args.c == ["1", "-1"] and args.func is cli.cmd_offtarget
