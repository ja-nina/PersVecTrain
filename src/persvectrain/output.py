"""Output layout of a training run, PersVecFast-compatible:

<out>/trained/<run_name>/{train.json, steps.jsonl} and .../persona/<model>/bf16/ with per trait
{slug}.pt, {slug}.json, {slug}_responses.json, {slug}_raw.pt, vllm_hook/{slug}_L{layer}[_random].pt.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import persvecfast
import torch
from persvecfast.extract import random_control, vector_file
from persvecfast.util import read_json, save_pt, write_json

from persvectrain.train import TrainConfig

REPO = Path(__file__).resolve().parents[2]
PERSVECFAST_REPO = Path(persvecfast.__file__).resolve().parents[2]


def source_dir(cfg: TrainConfig) -> Path:
    """PersVecFast's vector dir under the source out root."""
    return Path(cfg.source) / "persona" / cfg.model / "bf16"


def run_root(out: Path, run_name: str) -> Path:
    return Path(out) / "trained" / run_name


def run_dir(out: Path, run_name: str, model: str) -> Path:
    return run_root(out, run_name) / "persona" / model / "bf16"


def _inside(path: Path, parent: Path) -> bool:
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


def prepare_run(cfg: TrainConfig) -> Path:
    """Create and return the run's vector dir; refuses an existing run or a run dir inside the source dir."""
    d = run_dir(cfg.out, cfg.run_name, cfg.model)
    if _inside(d, source_dir(cfg)):
        raise ValueError(f"run dir {d} is inside the source dir {source_dir(cfg)}")
    root = run_root(cfg.out, cfg.run_name)
    if root.exists():
        raise FileExistsError(f"run {root} exists; pick another --run-name")
    d.mkdir(parents=True)
    return d


def write_vectors(dir: Path, slug: str, layer: int, trained: torch.Tensor, src: Path, run_name: str,
                  seed: int = 0) -> None:
    """Write one trait's files into the run dir `dir`; `{slug}.pt` last (it marks the trait done).

    Never writes into `src`. The random twin is random_control(trained, seed + layer), as PersVecFast does.
    """
    dir, src = Path(dir), Path(src)
    if _inside(dir, src):
        raise ValueError(f"refusing to write into the source dir {src}")
    for name in (f"{slug}_responses.json", f"{slug}_raw.pt"):
        if (src / name).exists():
            shutil.copy2(src / name, dir / name)
    source_vecs = torch.load(src / f"{slug}.pt")
    v = trained.detach().float().cpu().to(source_vecs[layer].dtype).clone()
    meta = read_json(src / f"{slug}.json") if (src / f"{slug}.json").exists() else {}
    meta["trained"] = {"run_name": run_name, "layer": layer, "source_norm": float(source_vecs[layer].float().norm()),
                       "trained_norm": float(v.float().norm())}
    write_json(dir / f"{slug}.json", meta)
    save_pt(vector_file(dir, slug, layer), {"dir": v.clone()})
    save_pt(vector_file(dir, slug, layer, random=True), {"dir": random_control(v, seed + layer)})
    out_vecs = dict(source_vecs)
    out_vecs[layer] = v.clone()
    save_pt(dir / f"{slug}.pt", out_vecs)


def git_commit(repo: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def config_dict(cfg: TrainConfig) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}


def build_train_json(cfg: TrainConfig, sources: dict, layer_type: str, dropped_traits: dict[str, str],
                     splits: dict) -> dict:
    return {
        "config": config_dict(cfg),
        "seed": cfg.seed,
        "persvectrain_commit": git_commit(REPO),
        "persvecfast_commit": git_commit(PERSVECFAST_REPO),
        "sources": sources,
        "layer_type": layer_type,
        "dropped_traits": dict(dropped_traits),
        "splits": splits,
    }
