"""Slurm scripts: syntax, and the argument extraction shared by run.sbatch (slurm/lib.sh)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[1] / "slurm"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def _bash(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, "-c", code, "x", *args], capture_output=True, text=True, cwd=SLURM.parent)


@pytest.mark.parametrize("name", ["run.sbatch", "train.sbatch", "sweep.sh", "lib.sh"])
def test_script_syntax(name):
    r = subprocess.run([BASH, "-n", str(SLURM / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_arg_value_extraction():
    code = 'source slurm/lib.sh; pvt_arg_value --out "$@"'
    r = _bash(code, "train", "--model", "m", "--out", "/o dir/x", "--run-name", "r1")
    assert r.returncode == 0 and r.stdout == "/o dir/x"
    r = _bash('source slurm/lib.sh; pvt_arg_value --run-name "$@"', "train", "--out", "o", "--run-name", "r1")
    assert r.stdout == "r1"


def test_arg_value_missing_is_an_error():
    r = _bash('source slurm/lib.sh; pvt_arg_value --run-name "$@"', "train", "--out", "o")
    assert r.returncode != 0 and "--run-name" in r.stderr
    r = _bash('source slurm/lib.sh; pvt_arg_value --out "$@"', "train", "--out")  # flag without a value
    assert r.returncode != 0


def test_has_flag():
    assert _bash('source slurm/lib.sh; pvt_has_flag --no-free-gen "$@"', "train", "--no-free-gen").returncode == 0
    assert _bash('source slurm/lib.sh; pvt_has_flag --no-free-gen "$@"', "train", "--out", "o").returncode != 0


def test_arg_value_equals_form():
    r = _bash('source slurm/lib.sh; pvt_arg_value --out "$@"', "train", "--out=/o dir/x", "--run-name=r1")
    assert r.returncode == 0 and r.stdout == "/o dir/x"
    r = _bash('source slurm/lib.sh; pvt_arg_value --run-name "$@"', "train", "--out=o", "--run-name=r1")
    assert r.stdout == "r1"
