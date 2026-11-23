from __future__ import annotations

import json
import math
import os

import pytest

from persvectrain import analysis as an
from persvectrain import cli
from persvectrain.output import run_root
from tests.test_cli import MODEL, _fake_reference, _fake_trait, _train_args, tiny  # noqa: F401

QIDS = ["q0", "q1", "q2", "q3"]


def _ans(correct, tokens):
    return {"qid": QIDS, "correct": correct, "tokens": tokens, "accuracy": sum(correct) / len(correct),
            "mean_tokens": sum(tokens) / len(tokens), "cut": [t >= 150 for t in tokens]}


def _judge(score, coh):
    return {"qid": QIDS, "score": score, "coherence": coh, "score_mean": sum(score) / 4,
            "coherence_mean": sum(coh) / 4}


def _run():
    unst = _ans([1, 1, 0, 0], [100, 100, 100, 100])
    bench_trait = {
        "trained": {"c=+1": _ans([1, 1, 1, 0], [110, 90, 100, 100]), "c=-1": _ans([1, 0, 0, 0], [100] * 4)},
        "source_full": {"c=+1": _ans([0, 0, 0, 0], [200, 50, 100, 100]), "c=-1": _ans([1, 1, 1, 1], [100] * 4)},
        "trained_random": {c: {**_ans([0, 1, 1, 0], [100, 100, 200, 100]), "direction": [0, 1, 0, 1]}
                           for c in ("c=+1", "c=-1")},
    }
    pj = {"unsteered": _judge([10, 10, 10, 10], [90, 90, 90, 90]),
          "trained": {"c=+1": _judge([80, 80, 80, 80], [70, 70, 70, 70]),
                      "c=-1": _judge([20, 20, 20, 20], [80, 80, 80, 80])},
          "source_full": {"c=+1": _judge([70, 70, 70, 70], [60, 60, 60, 60]),
                          "c=-1": _judge([30, 30, 30, 30], [80, 80, 80, 80])}}
    kinds = {"trained": {"c=+1": {"margins": [2.0, 3.0], "kls": [0.1, 0.1], "margin_mean": 2.5, "kl_mean": 0.1}},
             "source_full": {"c=+1": {"margins": [1.0, 1.0], "kls": [0.3, 0.5], "margin_mean": 1.0,
                                      "kl_mean": 0.4}}}
    return {"name": "r", "config": {"lambda_kl": 5.0, "seed": 0, "mode": "solo"},
            "eval": {"traits": {"evil": {"kinds": kinds, "persona_judge": pj}}},
            "bench": {"unsteered": unst, "traits": {"evil": bench_trait}}}


def _get(rows, metric, c, against="source"):
    (r,) = [r for r in rows if r["metric"] == metric and r["c"] == c and r["against"] == against]
    return r


def test_rows_are_oriented_so_positive_means_trained_better():
    rows = an.comparisons([_run()])
    assert _get(rows, "bench_acc", "c=+1")["diff"] == pytest.approx(0.75)
    assert _get(rows, "bench_acc", "c=-1")["diff"] == pytest.approx(-0.75)
    assert _get(rows, "bench_acc", "c=+1", "unsteered")["diff"] == pytest.approx(0.25)
    # length shift: source |200-100|+|50-100| = 150 over 4, trained 20 over 4 -> trained shifts less by 32.5
    assert _get(rows, "bench_len", "c=+1")["diff"] == pytest.approx(32.5)
    # judge: +1 more trait is better, -1 less trait is better (trained 20 vs source 30 -> +10)
    assert _get(rows, "judge_trait", "c=+1")["diff"] == pytest.approx(10)
    assert _get(rows, "judge_trait", "c=-1")["diff"] == pytest.approx(10)
    assert _get(rows, "judge_coherence", "c=+1")["diff"] == pytest.approx(10)
    assert _get(rows, "judge_coherence", "c=+1", "unsteered")["diff"] == pytest.approx(-20)
    assert not [r for r in rows if r["metric"] == "judge_trait" and r["against"] == "unsteered"]
    assert _get(rows, "tf_margin", "c=+1")["diff"] == pytest.approx(1.5)
    assert _get(rows, "tf_kl", "c=+1")["diff"] == pytest.approx(0.3)
    assert all(r["lambda_kl"] == 5.0 and r["trait"] == "evil" for r in rows)


def test_cut_and_uncut_accuracy():
    rows = an.comparisons([_run()])
    cut = _get(rows, "bench_cut", "c=+1")  # source cut 1 of 4, trained none
    assert cut["diff"] == pytest.approx(0.25) and cut["against_value"] == pytest.approx(0.25)
    unc = _get(rows, "bench_acc_uncut", "c=+1")  # questions 1-3: trained 1,1,0 vs source 0,0,0
    assert unc["n"] == 3 and unc["diff"] == pytest.approx(2 / 3)


def test_random_directions():
    rows = an.random_direction_rows(_run())
    # direction 0 = questions 0, 2: random 0, 1 vs unsteered 1, 0 -> 0; direction 1 = questions 1, 3: 1, 0 vs 1, 0
    d = {(r["direction"], r["c"]): r for r in rows}
    assert d[(0, "c=+1")]["diff"] == pytest.approx(0) and d[(0, "c=+1")]["cut"] == pytest.approx(0.5)
    assert d[(1, "c=-1")]["diff"] == pytest.approx(0) and d[(1, "c=-1")]["n"] == 2
    summ = an.random_direction_summary(rows)
    assert len(summ) == 4 and all(x["cells"] == 1 for x in summ)
    assert "each random direction" in an.format_report(an.comparisons([_run()]), None, 2.0, rows)


def test_worse_and_summary():
    rows = an.comparisons([_run()])
    w = an.worse(rows, z=1.0)
    # nothing is cut at c=-1, so the uncut accuracy is the same cell
    assert sorted((r["metric"], r["c"]) for r in w) == [("bench_acc", "c=-1"), ("bench_acc_uncut", "c=-1")]
    assert w[0]["z"] == pytest.approx(-0.75 / 0.25)
    s = {(x["metric"], x["against"], x["c_sign"]): x for x in an.summary(rows, z=1.0)}
    assert s[("bench_acc", "source", "-")]["worse"] == 1 and s[("bench_acc", "source", "+")]["better"] == 1
    assert "trained is worse than source" in an.format_report(rows, None, 1.0)


def test_paired_skips_none_and_checks_length():
    assert an.paired([1, None, 3], [0, 0, None]) == (1.0, None, 1)
    with pytest.raises(ValueError):
        an.paired([1], [1, 2])


def test_old_format_block_is_skipped(capsys):
    r = _run()
    del r["bench"]["traits"]["evil"]["source_full"]["c=+1"]["qid"]
    rows = an.comparisons([r])
    assert not [x for x in rows if x["metric"] == "bench_acc" and x["c"] == "c=+1" and x["against"] == "source"]
    assert _get(rows, "bench_acc", "c=-1")["diff"] == pytest.approx(-0.75)
    assert "old format" in capsys.readouterr().err


def test_unpaired_question_ids_raise():
    r = _run()
    r["bench"]["traits"]["evil"]["source_full"]["c=+1"]["qid"] = ["x"] * 4
    with pytest.raises(ValueError, match="not paired"):
        an.comparisons([r])


@pytest.fixture
def out_two_runs(tiny, tmp_path):
    out = tmp_path / "out"
    src = out / "persona" / MODEL / "bf16"
    for t in ("assertive", "calm"):
        _fake_trait(src, t)
    _fake_reference(out)
    for run, lam in (("run1", "1"), ("run2", "10")):
        args = _train_args(out, "--no-free-gen", "--no-judge", "--lambda-kl", lam)
        args[args.index("Evil")] = "Calm"
        args[args.index("run1")] = run
        assert cli.main(args) == 0
    # a bench.json for run2 only (run1 is "not benched yet")
    b = {"unsteered": _ans([1, 0, 1, 0], [100] * 4),
         "traits": {s: {"trained": {"c=+1": _ans([1, 1, 1, 0], [100] * 4), "c=-1": _ans([1, 0, 1, 0], [100] * 4)},
                        "source_full": {"c=+1": _ans([0, 0, 1, 0], [150] * 4),
                                        "c=-1": _ans([0, 0, 0, 0], [100] * 4)}}
                    for s in ("assertive", "calm")}}
    (run_root(out, "run2") / "bench.json").write_text(json.dumps(b), encoding="utf-8")
    # persona judge and 100-question free answers for run2
    ev_path = run_root(out, "run2") / "eval.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    for s, t in ev["traits"].items():
        t["persona_judge"] = {"unsteered": _judge([10, 10, 12, 10], [90, 92, 90, 88]),
                              **{k: {"c=+1": _judge([80, 70, 75, 60 + i], [70, 75, 72, 71]),
                                     "c=-1": _judge([20, 25, 15, 30 - i], [80, 82, 78, 85])}
                                 for i, k in enumerate(("trained", "source_full", "source_at_trained_norm",
                                                        "trained_random"))}}
        t["free_gen"] = {"unsteered": _ans([1, 0, 0, 0], [90] * 4),
                         "trained": {"c=+1": _ans([1, 1, 0, 0], [90] * 4), "c=-1": _ans([0, 1, 0, 0], [90] * 4)},
                         "source_full": {"c=+1": _ans([0, 1, 0, 0], [99] * 4), "c=-1": _ans([0, 0, 0, 0], [90] * 4)}}
    ev_path.write_text(json.dumps(ev), encoding="utf-8")
    return out


def test_analysis_on_real_runs(out_two_runs, capsys):
    out = out_two_runs
    assert an.discover_runs(out) == ["run1", "run2"]
    runs = [an.load_run(out, r) for r in ("run1", "run2")]
    assert runs[0]["bench"] is None and runs[1]["bench"] is not None
    rows = an.comparisons(runs)
    assert {r["metric"] for r in rows} == set(an.METRICS)
    assert {r["metric"] for r in rows if r["run"] == "run1"} == {"tf_margin", "tf_kl"}
    assert {r["against"] for r in rows if r["metric"] == "tf_margin"} == {"source", "srcN", "random"}
    rot = an.rotation(out, ["run1", "run2"])
    assert len(rot["per_trait"]) == 4
    for r in rot["per_trait"]:
        assert r["angle_start_deg"] == pytest.approx(math.degrees(math.acos(r["cos_trained_start"])), abs=1e-6)
    assert set(rot["trained_cos"]) == {"assertive", "calm"}
    json.dumps(rot["analyse"])  # tensors removed: serialisable
    assert cli.main(["analysis", "--out", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "rotation per run" in txt and "bench_acc" in txt and "run2" in txt
    assert cli.main(["analysis", "--out", str(out), "--runs", "run2", "--no-vectors"]) == 0
    assert "rotation" not in capsys.readouterr().out


def test_notebook_runs_on_real_runs(out_two_runs, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=True)  # plt.show() must not open (and wait on) a window
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PVT_OUT", str(out_two_runs))
    nb = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks" / "analysis.ipynb", as_version=4)
    ns: dict = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "<cell>", "exec"), ns)
    assert ns["RUNS"] == ["run1", "run2"]


def test_run_notebook_writes_pngs(out_two_runs, tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv("PVT_OUT", str(out_two_runs))
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_notebook.py"
    spec = importlib.util.spec_from_file_location("run_notebook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    figs = tmp_path / "figs"
    assert mod.main(["--dir", str(figs)]) == 0
    pngs = sorted(figs.glob("*.png"))
    assert len(pngs) >= 8 and all(p.stat().st_size > 1000 for p in pngs)
    rep = (figs / "report.txt").read_text(encoding="utf-8")
    assert "trained worse than source" in rep and "rotation" in rep.lower()
