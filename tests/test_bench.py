from __future__ import annotations

import json

import pytest
from persvecfast.models import request_seed

from persvectrain import bench as bm
from persvectrain import cli
from persvectrain import freegen as fgm
from tests.test_freegen import HELD, KINDS, _fake, _reqs, run  # noqa: F401  (the finished-run fixture)

BQ = ["b0", "b1", "b2", "b3"]


def _bench_file(tmp_path, ids=BQ):
    rows = [{"id": q, "question": f"Bench {q}?", "options": [f"o{j}" for j in range(4)], "answer": "B"} for q in ids]
    p = tmp_path / "bench.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _reply(r):
    # unsteered and source right on every question; trained right on b0, b1 only; random right on b0
    slug, kind, c, q = r.key
    right = kind in ("unsteered", "source_full") or (kind == "trained" and q in ("b0", "b1")) or \
        (kind == "trained_random" and q == "b0")
    return ("ANSWER: B" if right else "ANSWER: A", 7, "stop")


def test_bench_requests_results_and_diffs(run, monkeypatch, tmp_path):
    engine, opened = _fake(monkeypatch, _reply)
    before = {p: p.read_bytes() for p in run["root"].glob("*.json")}
    res = bm.run_bench(run["out"], "run1", questions=_bench_file(tmp_path), max_tokens=64)
    reqs = _reqs(engine)
    assert opened == [(run["cfg"].model, "steer")]
    assert len(reqs) == len(BQ) * (1 + 4 * 2 * 2)  # unsteered + 2 traits x 4 kinds x 2 c
    by_key = {r.key: r for r in reqs}
    for q in BQ:
        assert by_key[("-", "unsteered", 0.0, q)].steer is None
        seeds = {by_key[(s, k, 1.0, q)].seed for s in ("assertive", "calm") for k in KINDS}
        assert seeds == {request_seed(0, "freegen", 1.0, q)}  # paired across kinds and traits
    # random rotation: question i uses direction k = i % 5
    assert by_key[("calm", "trained_random", -1.0, "b3")].steer["vector_path"].endswith("calm_trained_random_k3.pt")
    # eval.json / train.json untouched; bench.json written
    assert all(p.read_bytes() == b for p, b in before.items())
    saved = json.loads((run["root"] / bm.OUT_FILE).read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(res)) and res["n"] == 4
    assert res["unsteered"]["accuracy"] == 1.0
    t = res["traits"]["assertive"]
    assert t["trained"]["c=+1"]["accuracy"] == 0.5 and t["source_full"]["c=-1"]["accuracy"] == 1.0
    d = t["diffs"]["c=+1"]
    assert d["trained-unsteered"]["mean"] == pytest.approx(-0.5)
    assert d["source-unsteered"]["mean"] == 0.0 and d["trained-source"]["mean"] == pytest.approx(-0.5)
    assert d["random-unsteered"]["mean"] == pytest.approx(-0.75)
    assert "trained-unst" in bm.format_bench(res)


def test_bench_resumes_from_saved_answers(run, monkeypatch, tmp_path):
    q = _bench_file(tmp_path)
    engine, _ = _fake(monkeypatch, _reply)
    bm.run_bench(run["out"], "run1", questions=q, max_tokens=64)
    n_first = len(_reqs(engine))
    engine2, opened2 = _fake(monkeypatch, _reply)
    res = bm.run_bench(run["out"], "run1", questions=q, max_tokens=64)
    assert n_first > 0 and _reqs(engine2) == [] and opened2 == []  # everything reused, no engine
    assert res["traits"]["calm"]["trained"]["c=+1"]["accuracy"] == 0.5
    # a different max_tokens is a different request: regenerated
    engine3, _ = _fake(monkeypatch, _reply)
    bm.run_bench(run["out"], "run1", questions=q, max_tokens=32)
    assert len(_reqs(engine3)) == n_first


def test_bench_n_and_overlap_with_the_kl_reference_refused(run, monkeypatch, tmp_path):
    engine, _ = _fake(monkeypatch, _reply)
    res = bm.run_bench(run["out"], "run1", questions=_bench_file(tmp_path), n=2, max_tokens=64)
    assert res["n"] == 2 and res["unsteered"]["qid"] == ["b0", "b1"]
    with pytest.raises(SystemExit, match="reference split"):
        bm.run_bench(run["out"], "run1", questions=_bench_file(tmp_path, ["b0", HELD[0]]), max_tokens=64)


def test_bench_unfinished_run_does_nothing(run, monkeypatch, tmp_path, capsys):
    meta = json.loads((run["root"] / "train.json").read_text(encoding="utf-8"))
    meta["finished"] = False
    (run["root"] / "train.json").write_text(json.dumps(meta), encoding="utf-8")
    assert bm.run_bench(run["out"], "run1", questions=_bench_file(tmp_path)) is None
    assert "did not finish" in capsys.readouterr().out


def test_cli_bench_and_report_section(run, monkeypatch, tmp_path, capsys):
    _fake(monkeypatch, _reply)
    rc = cli.main(["bench", "--out", str(run["out"]), "--run-name", "run1", "--questions",
                   str(_bench_file(tmp_path)), "--max-tokens", "64"])
    assert rc == 0 and (run["root"] / bm.OUT_FILE).exists()
    capsys.readouterr()
    cli.main(["report", "--out", str(run["out"]), "--runs", "run1"])
    assert "bench run1: 4 SuperGPQA questions" in capsys.readouterr().out


def test_bench_file_in_repo_is_new_questions():
    """data/supergpqa_bench500.jsonl: 500 questions, none in the KL pool."""
    bench = [json.loads(l)["id"] for l in bm.BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = {json.loads(l)["id"] for l in fgm.POOL.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert len(bench) == len(set(bench)) == 500 and not set(bench) & pool


def test_bench_length_control_vector(run, monkeypatch, tmp_path):
    """source_at_trained_norm: the source direction at exactly the trained vector's length; its diffs are reported."""
    import torch
    _fake(monkeypatch, _reply)
    res = bm.run_bench(run["out"], "run1", questions=_bench_file(tmp_path), max_tokens=64)
    d = run["root"] / "freegen_vectors"
    for slug in ("assertive", "calm"):
        nm = torch.load(d / f"{slug}_{fgm.NORM_MATCHED}.pt")["dir"]
        t, sf = torch.load(d / f"{slug}_trained.pt")["dir"], torch.load(d / f"{slug}_source_full.pt")["dir"]
        assert float(nm.norm()) == pytest.approx(float(t.norm()), rel=1e-5)
        assert float(torch.nn.functional.cosine_similarity(nm, sf, dim=0)) == pytest.approx(1.0, abs=1e-5)
    assert res["traits"]["calm"]["diffs"]["c=-1"]["trained-normsrc"]["mean"] == pytest.approx(0.5)  # fake: srcN always wrong
    assert "trained-srcN" in bm.format_bench(res)
