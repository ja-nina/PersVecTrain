from __future__ import annotations

import json
from pathlib import Path

from persvectrain import cli
from persvectrain.report import format_table, mean_trained_margin, pick_run


def _trait(passed: bool, margin: float = 1.0, failed=None) -> dict:
    kinds = {"trained": {"c=+1": {"margin_mean": margin}, "c=-1": {"margin_mean": margin}}}
    return {"kinds": kinds,
            "pass": {"checks": {}, "failed": failed if failed is not None else ([] if passed else ["kl@c=+1"]),
                     "passed": passed}}


def _eval(**traits: dict) -> dict:
    return {"traits": traits, "summary": {k: v["pass"]["passed"] for k, v in traits.items()}}


def test_report_picks_most_passing():
    evals = {"a": _eval(x=_trait(True), y=_trait(False)),
             "b": _eval(x=_trait(True), y=_trait(True)),
             "c": _eval(x=_trait(False), y=_trait(False))}
    assert pick_run(evals) == "b"


def test_report_none_passes():
    evals = {"a": _eval(x=_trait(False)), "b": _eval(x=_trait(False))}
    assert pick_run(evals) is None
    assert pick_run({}) is None


def test_report_tie_broken_by_mean_trained_margin():
    t = {"c=+1": {"margin_mean": 1.0}, "c=-1": {"margin_mean": 3.0}}
    e = _eval(x=_trait(True))
    e["traits"]["x"]["kinds"]["trained"] = t
    assert mean_trained_margin(e) == 2.0
    evals = {"low": _eval(x=_trait(True, margin=0.5)), "high": _eval(x=_trait(True, margin=2.0)),
             "mid": _eval(x=_trait(True, margin=1.0))}
    assert pick_run(evals) == "high"


def test_report_table_lists_failed_checks():
    text = format_table({"a": _eval(x=_trait(True), y=_trait(False, failed=["kl@c=+1", "persona@c=-1"]))})
    assert "PASS" in text and "FAIL (kl@c=+1, persona@c=-1)" in text and "x" in text and "y" in text


def _write_eval(out: Path, run: str, e: dict) -> None:
    root = out / "trained" / run
    root.mkdir(parents=True)
    (root / "eval.json").write_text(json.dumps(e), encoding="utf-8")


def test_report_cli_missing_eval_json(tmp_path, capsys):
    _write_eval(tmp_path, "good", _eval(x=_trait(True)))
    assert cli.main(["report", "--out", str(tmp_path), "--runs", "nope", "good"]) == 0
    out = capsys.readouterr().out
    assert "nope: no eval.json" in out and "picked: good" in out


def test_report_cli_no_run_passes(tmp_path, capsys):
    _write_eval(tmp_path, "bad", _eval(x=_trait(False)))
    assert cli.main(["report", "--out", str(tmp_path), "--runs", "bad"]) == 0
    assert "no run passes" in capsys.readouterr().out


def _pending(ev: dict) -> dict:
    ev["free_gen"] = {"enabled": True, "status": "pending"}
    return ev


def test_report_pending_run_never_picked_even_if_passed():
    evals = {"pend": _pending(_eval(x=_trait(True), y=_trait(True))), "done": _eval(x=_trait(True))}
    assert pick_run(evals) == "done"
    assert pick_run({"pend": _pending(_eval(x=_trait(True)))}) is None


def test_report_table_shows_pending_per_trait():
    text = format_table({"pend": _pending(_eval(x=_trait(True), y=_trait(False))), "done": _eval(x=_trait(True))})
    pend_row = [ln for ln in text.splitlines() if ln.startswith("pend")][0]
    assert pend_row.split().count("PENDING") == 2 and "PASS" not in pend_row
    assert "PENDING (FAIL: kl@c=+1)" in pend_row  # y already fails a teacher-forced check
    assert "PASS" in [ln for ln in text.splitlines() if ln.startswith("done")][0]


def test_report_disabled_run_is_reported_normally():
    ev = _eval(x=_trait(True))
    ev["free_gen"] = {"enabled": False, "status": "disabled"}
    assert pick_run({"off": ev}) == "off"
    assert "PASS" in format_table({"off": ev})


def test_report_cli_pending_not_picked(tmp_path, capsys):
    _write_eval(tmp_path, "pend", _pending(_eval(x=_trait(True))))
    assert cli.main(["report", "--out", str(tmp_path), "--runs", "pend"]) == 0
    out = capsys.readouterr().out
    assert "PENDING" in out and "no run passes" in out


def test_report_pending_run_shows_teacher_forced_failures():
    ev = _pending(_eval(x=_trait(True), y=_trait(False, failed=["kl@c=+1", "persona@c=-1"])))
    text = format_table({"pend": ev})
    row = [ln for ln in text.splitlines() if ln.startswith("pend")][0]
    assert "PENDING (FAIL: kl@c=+1, persona@c=-1)" in row and row.count("PENDING") == 2
    assert "PASS" not in row
    assert pick_run({"pend": ev}) is None
