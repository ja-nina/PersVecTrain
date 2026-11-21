"""Output layout of a training run, PersVecFast-compatible:

<out>/trained/<run_name>/{train.json, steps.jsonl} and .../persona/<model>/bf16/ with per trait
{slug}.pt, {slug}.json, {slug}_responses.json, {slug}_raw.pt, vllm_hook/{slug}_L{layer}[_random].pt.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import persvecfast
import torch
from persvecfast.extract import RANDOM_CONTROL_SEED, random_control, vector_file
from persvecfast.util import read_json, save_pt, write_json

from persvectrain.train import RefAnswer, TrainConfig

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


def write_vectors(dir: Path, slug: str, layer: int, trained: torch.Tensor, src: Path, run_name: str) -> None:
    """Write one trait's files into the run dir `dir`; `{slug}.pt` last (it marks the trait done).

    Never writes into `src`. The random twin is random_control(trained, RANDOM_CONTROL_SEED + layer) whatever the
    training seed, as PersVecFast and MA_Environments do.
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
    save_pt(vector_file(dir, slug, layer, random=True), {"dir": random_control(v, RANDOM_CONTROL_SEED + layer)})
    out_vecs = dict(source_vecs)
    out_vecs[layer] = v.clone()
    save_pt(dir / f"{slug}.pt", out_vecs)


def git_commit(repo: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def git_dirty(repo: Path) -> bool | None:
    """True when `git status --porcelain` lists anything (uncommitted or untracked files); None if unknown."""
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True,
                           timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return bool(r.stdout.strip())


def config_dict(cfg: TrainConfig) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}


def build_train_json(cfg: TrainConfig, sources: dict, layer_type: str | None, dropped_traits: dict[str, str],
                     splits: dict) -> dict:
    return {
        "config": config_dict(cfg),
        "seed": cfg.seed,
        "persvectrain_commit": git_commit(REPO),
        "persvecfast_commit": git_commit(PERSVECFAST_REPO),
        "persvectrain_dirty": git_dirty(REPO),
        "persvecfast_dirty": git_dirty(PERSVECFAST_REPO),
        "sources": sources,
        "layer_type": layer_type,
        "dropped_traits": dict(dropped_traits),
        "splits": splits,
    }


# ---------------------------------------------------------------- reading a run back


def config_from_train_json(meta: dict, out: Path) -> TrainConfig:
    """The run's TrainConfig from its train.json, with `out` the out root the run is read from."""
    kw = dict(meta["config"])
    kw["source"], kw["out"] = Path(kw["source"]), Path(out).resolve()
    return TrainConfig(**kw)


def load_reference(path: Path) -> list[RefAnswer]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [RefAnswer(id=r["id"], question=r["question"], answer_text=r["answer_text"], gold=r["gold"])
            for r in rows]


def load_trained(vec_dir: Path, slugs: list[str], layer: int) -> dict[str, torch.Tensor]:
    """The run's trained vectors from vllm_hook/{slug}_L{layer}.pt, as fp32."""
    out = {}
    for slug in slugs:
        path = vector_file(vec_dir, slug, layer)
        if not path.exists():
            raise SystemExit(f"trained vector {path} missing: training did not finish writing {slug}")
        out[slug] = torch.load(path)["dir"].float()
    return out
