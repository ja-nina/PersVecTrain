from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from persvectrain.output import build_train_json, prepare_run, run_dir, source_dir, write_vectors
from persvectrain.train import TrainConfig

HIDDEN = 16
LAYERS = 3


def _cfg(tmp_path: Path, **kw) -> TrainConfig:
    base = dict(model="tiny", layer=1, mode="solo", traits=["assertive"], source=tmp_path / "out",
                out=tmp_path / "out", run_name="r1")
    base.update(kw)
    return TrainConfig(**base)


def _fake_source(d: Path, slug: str = "assertive", with_json: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(3)
    (d / f"{slug}_responses.json").write_text(json.dumps({"positive": [], "negative": []}), encoding="utf-8")
    torch.save({"positive": {}, "negative": {}, "kept_questions": []}, d / f"{slug}_raw.pt")
    torch.save({l: torch.randn(HIDDEN, generator=g) for l in range(LAYERS)}, d / f"{slug}.pt")
    if with_json:
        (d / f"{slug}.json").write_text(json.dumps({"trait": "Assertive", "layer_scores": {"1": 0.5}}),
                                        encoding="utf-8")


def _hashes(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def test_run_dir_and_source_dir(tmp_path):
    cfg = _cfg(tmp_path)
    assert run_dir(cfg.out, "r1", "tiny") == tmp_path / "out" / "trained" / "r1" / "persona" / "tiny" / "bf16"
    assert source_dir(cfg) == tmp_path / "out" / "persona" / "tiny" / "bf16"


def test_prepare_run_creates_run_dir(tmp_path):
    cfg = _cfg(tmp_path)
    d = prepare_run(cfg)
    assert d == run_dir(cfg.out, cfg.run_name, cfg.model) and d.is_dir()


def test_prepare_run_refuses_existing_and_source(tmp_path):
    # existing run
    cfg = _cfg(tmp_path)
    (cfg.out / "trained" / "r1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        prepare_run(cfg)
    assert list((cfg.out / "trained" / "r1").iterdir()) == []

    # run dir inside the source vector dir
    src_root = tmp_path / "src"
    bad_out = src_root / "persona" / "tiny" / "bf16"
    cfg2 = _cfg(tmp_path, source=src_root, out=bad_out, run_name="r2")
    before = sorted(p for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        prepare_run(cfg2)
    assert sorted(p for p in tmp_path.rglob("*")) == before
    assert not (bad_out / "trained").exists()


def test_write_vectors_layout(tmp_path):
    src = tmp_path / "out" / "persona" / "tiny" / "bf16"
    _fake_source(src)
    before = _hashes(src)
    dst = tmp_path / "out" / "trained" / "r1" / "persona" / "tiny" / "bf16"
    dst.mkdir(parents=True)
    trained = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(9))

    write_vectors(dst, "assertive", 1, trained, src, "r1", seed=0)

    for name in ("assertive_responses.json", "assertive_raw.pt", "assertive.json", "assertive.pt",
                 "vllm_hook/assertive_L1.pt", "vllm_hook/assertive_L1_random.pt"):
        assert (dst / name).exists(), name
    source_vecs = torch.load(src / "assertive.pt")
    out_vecs = torch.load(dst / "assertive.pt")
    assert set(out_vecs) == set(source_vecs)
    assert torch.equal(out_vecs[1], trained)
    for l in (0, 2):
        assert torch.equal(out_vecs[l], source_vecs[l])
    hook = torch.load(dst / "vllm_hook" / "assertive_L1.pt")
    assert set(hook) == {"dir"} and torch.equal(hook["dir"], trained)
    rnd = torch.load(dst / "vllm_hook" / "assertive_L1_random.pt")
    assert set(rnd) == {"dir"}
    assert torch.isclose(rnd["dir"].norm(), trained.norm())
    assert not torch.allclose(rnd["dir"], trained)
    meta = json.loads((dst / "assertive.json").read_text(encoding="utf-8"))
    assert meta["trait"] == "Assertive"
    assert meta["trained"]["run_name"] == "r1" and meta["trained"]["layer"] == 1
    assert meta["trained"]["source_norm"] == pytest.approx(float(source_vecs[1].norm()))
    assert meta["trained"]["trained_norm"] == pytest.approx(float(trained.norm()))
    assert _hashes(src) == before


def test_write_vectors_without_source_json(tmp_path):
    src = tmp_path / "src"
    _fake_source(src, with_json=False)
    dst = tmp_path / "dst"
    dst.mkdir()
    write_vectors(dst, "assertive", 1, torch.ones(HIDDEN), src, "r1")
    meta = json.loads((dst / "assertive.json").read_text(encoding="utf-8"))
    assert set(meta) == {"trained"}


def test_train_json_has_config_seed_commits(tmp_path):
    cfg = _cfg(tmp_path)
    d = build_train_json(cfg, sources={"assertive": {"vectors": "x"}, "reference": "ref.jsonl"},
                         layer_type="full_attention", dropped_traits={"evil": "missing"},
                         splits={"assertive": {"train": [0], "held": [1]}, "reference": {"train": ["a"],
                                                                                           "held": ["b"]}})
    for key in ("config", "seed", "persvectrain_commit", "persvecfast_commit", "sources", "layer_type",
                "dropped_traits", "splits"):
        assert key in d, key
    assert d["seed"] == cfg.seed
    assert d["config"]["source"] == str(cfg.source) and d["config"]["layer"] == 1
    assert isinstance(d["persvectrain_commit"], str) and isinstance(d["persvecfast_commit"], str)
    json.dumps(d)  # serialisable
