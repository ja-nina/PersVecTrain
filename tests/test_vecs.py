from __future__ import annotations

import json

import pytest
import torch

from persvectrain import cli
from persvectrain import vecs as vm
from tests.conftest import CharTokenizer, make_tiny_model
from tests.test_cli import MODEL, _fake_reference, _fake_trait, _train_args, tiny  # noqa: F401


def test_trait_change_parts():
    s0 = torch.tensor([2.0, 0.0, 0.0])
    t = torch.tensor([3.0, 1.0, 0.0])  # delta (1, 1, 0): along s0 +1 (= +0.5 |s0|), orthogonal 1
    sf = torch.tensor([2.0, 0.0, 0.2])
    c = vm.trait_change(s0, t, sf)
    assert c["norm_ratio"] == pytest.approx(10 ** 0.5 / 2)
    assert c["delta_along_start_rel"] == pytest.approx(0.5) and c["delta_perp_rel"] == pytest.approx(0.5)
    assert c["split_noise_rel"] == pytest.approx(0.1)


def test_shared_finds_a_common_direction():
    g = torch.Generator().manual_seed(0)
    common = torch.randn(256, generator=g)
    same = {f"t{i}": common + 0.3 * torch.randn(256, generator=g) for i in range(5)}
    indep = {f"t{i}": torch.randn(256, generator=g) for i in range(5)}
    a, b = vm.shared(same), vm.shared(indep)
    assert a["sv_share"][0] > 0.8 and a["mean_cos"] > 0.8 and all(c > 0.9 for c in a["u1_cos"].values())
    assert b["sv_share"][0] < 0.4 and b["mean_abs_cos"] < 0.2
    assert vm.random_abs_cos(256) == pytest.approx((2 / (3.14159265 * 256)) ** 0.5, rel=1e-4)


def test_geometry_finds_the_high_variance_direction():
    g = torch.Generator().manual_seed(1)
    H = torch.randn(2000, 64, generator=g)
    H[:, 3] *= 20  # one dominant direction
    e3, e5 = torch.zeros(64), torch.zeros(64)
    e3[3], e5[5] = 1.0, 1.0
    geo = vm.geometry(H, {"hi": e3, "lo": e5})
    assert geo["hi"]["top1"] > 0.95 and geo["lo"]["top1"] < 0.05
    assert geo["hi"]["var_vs_random"] > 20 and geo["lo"]["var_vs_random"] < 1.5


@pytest.fixture
def two_runs(tiny, tmp_path):
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
    return out


def test_cli_vecs_without_and_with_model(two_runs, monkeypatch, capsys):
    out = two_runs
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert cli.main(["vecs", "--out", str(out), "--runs", "run1", "run2"]) == 0
    text = capsys.readouterr().out
    assert "shared change" in text and "split noise" in text and "run1 vs run2" in text
    path = out / "trained" / "analysis" / "vecs_run1_run2.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    r = res["runs"]["run1"]
    assert set(r["per_trait"]) == {"assertive", "calm"} and len(r["delta_perp"]["sv_share"]) == 2
    assert set(res["across_runs"]["run1 vs run2"]["delta_cos"]) == {"assertive", "calm"}
    bm = res["across_runs"]["run1 vs run2"]["b_minus_a"]
    assert set(bm["rel"]) == {"assertive", "calm"} and len(bm["shared_perp"]["sv_share"]) == 2
    assert "second - first" in text
    after = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert set(after) - set(before) == {path} and all(after[p] == b for p, b in before.items())  # reads only

    monkeypatch.setattr(cli, "load_model", lambda key: (make_tiny_model(), CharTokenizer()))
    assert cli.main(["vecs", "--out", str(out), "--runs", "run1", "--model", "--n", "3"]) == 0
    text = capsys.readouterr().out
    assert "logit lens u1 up" in text and "residual geometry" in text
    r = json.loads((out / "trained" / "analysis" / "vecs_run1.json").read_text(encoding="utf-8"))["runs"]["run1"]
    assert len(r["logit_lens"]["u1"]["up"]) == 15 and "assertive:delta" in r["geometry"]
