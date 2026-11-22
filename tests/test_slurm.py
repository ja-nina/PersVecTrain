"""Slurm scripts: syntax, and the argument extraction shared by run.sbatch (slurm/lib.sh)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[1] / "slurm"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def _bash(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, "-c", code, "x", *args], capture_output=True, text=True, cwd=SLURM.parent)


@pytest.mark.parametrize("name", ["run.sbatch", "train.sbatch", "sweep.sh", "lib.sh", "reeval.sbatch", "ablate.sh",
                                  "bench.sbatch", "vecs.sbatch", "pvt.sbatch", "followup.sh"])
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


def test_sweep_exports_lib_path_from_any_cwd(tmp_path):
    """sweep.sh run from another dir: every sbatch sees PVT_LIB = this checkout's slurm/lib.sh."""
    log = tmp_path / "sbatch.log"
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    # a fake sbatch (exported function: works on Windows paths too, unlike a PATH entry)
    code = (f'sbatch() {{ echo "$PVT_LIB" >> "{log.as_posix()}"; echo 42; }}; export -f sbatch; '
            f'bash "{(SLURM / "sweep.sh").as_posix()}" m 12')
    r = subprocess.run([BASH, "-c", code], capture_output=True, text=True, cwd=elsewhere)
    assert r.returncode == 0, r.stderr
    libs = log.read_text(encoding="utf-8").split()
    assert len(libs) == 5 and len(set(libs)) == 1
    r = subprocess.run([BASH, "-c", f'[ -f "{libs[0]}" ] && cmp -s "{libs[0]}" "{(SLURM / "lib.sh").as_posix()}"'])
    assert r.returncode == 0, libs[0]


def test_run_sbatch_runs_persona_after_freegen_unless_no_judge():
    text = " ".join((SLURM / "run.sbatch").read_text(encoding="utf-8").split())
    assert text.index("persvectrain freegen") < text.index("persvectrain persona")
    assert '-m persvectrain persona --out "$OUT" --run-name "$RUN_NAME"' in text
    assert 'pvt_has_flag --no-judge "$@"' in text and 'pvt_has_flag --no-free-gen "$@"' in text


def test_ablate_submits_reeval_for_existing_and_train_for_missing(tmp_path):
    out = tmp_path / "out"
    (out / "trained" / "solo_kl1").mkdir(parents=True)
    (out / "trained" / "solo_kl1" / "train.json").write_text('{"finished": true}', encoding="utf-8")
    stub = tmp_path / "stub.sh"
    log = tmp_path / "calls.log"
    stub.write_bytes(f'#!/bin/bash\necho "$@" >> "{log.as_posix()}"\necho 7\n'.encode())
    env = {**os.environ, "SBATCH": stub.as_posix(), "OUT": out.as_posix(), "SOURCE": "/src",
           "LAMBDAS": "1 3", "KL_TOKENS": "512 2048"}
    r = subprocess.run([BASH, str(SLURM / "ablate.sh"), "m", "12", "-p", "x"], capture_output=True, text=True,
                       cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 5  # reference 2048, reeval kl1, run kl3, run kl1_t2048, run kl3_t2048
    refs = [c for c in calls if "train.sbatch reference" in c]
    assert len(refs) == 1 and "--max-new-tokens 2048" in refs[0] and "--model m" in refs[0]
    reeval = [c for c in calls if "reeval.sbatch" in c]
    assert len(reeval) == 1 and "--run-name solo_kl1" in reeval[0] and "--dependency" not in reeval[0]
    runs = {c.split("--run-name ")[1].split()[0]: c for c in calls if "run.sbatch" in c}
    assert set(runs) == {"solo_kl3", "solo_kl1_t2048", "solo_kl3_t2048"}
    assert "--dependency" not in runs["solo_kl3"] and "--kl-tokens" not in runs["solo_kl3"]
    for name in ("solo_kl1_t2048", "solo_kl3_t2048"):
        assert "--dependency=afterok:7" in runs[name] and "--kl-tokens 2048" in runs[name]
    assert "--lambda-kl 3" in runs["solo_kl3"] and "--beta 0.01" in runs["solo_kl3"] and "-p x" in runs["solo_kl3"]
    assert "--per-trait" in r.stdout and "solo_kl1_t2048" in r.stdout


def _ablate(tmp_path, extra_env=None, setup=None):
    out = tmp_path / "out"
    (out / "trained").mkdir(parents=True, exist_ok=True)
    for run, fin in (setup or {}).items():
        (out / "trained" / run).mkdir(parents=True, exist_ok=True)
        (out / "trained" / run / "train.json").write_text('{"finished": %s}' % fin, encoding="utf-8")
    stub = tmp_path / "stub.sh"
    log = tmp_path / "calls.log"
    stub.write_bytes(f'#!/bin/bash\necho "$@" >> "{log.as_posix()}"\necho 7\n'.encode())
    env = {**os.environ, "SBATCH": stub.as_posix(), "OUT": out.as_posix(), "SOURCE": "/src", "LAMBDAS": "1",
           "SQUEUE": "true", **(extra_env or {})}
    r = subprocess.run([BASH, str(SLURM / "ablate.sh"), "m", "12"], capture_output=True, text=True, cwd=tmp_path,
                       env=env)
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return r, calls, out


def test_ablate_reference_present_no_reference_call_no_dependency(tmp_path):
    (tmp_path / "out" / "trained" / "reference").mkdir(parents=True)
    (tmp_path / "out" / "trained" / "reference" / "m_pool260_t2048.jsonl").write_text("", encoding="utf-8")
    r, calls, _ = _ablate(tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(calls) == 2 and not any("train.sbatch reference" in c for c in calls) and not any("--dependency" in c for c in calls)


def test_ablate_empty_train_extra_has_no_beta(tmp_path):
    r, calls, _ = _ablate(tmp_path, {"TRAIN_EXTRA": ""})
    assert r.returncode == 0, r.stderr
    assert calls and not any("--beta" in c for c in calls)


def test_ablate_unfinished_run_skipped_with_warning(tmp_path):
    r, calls, _ = _ablate(tmp_path, setup={"solo_kl1": "false"})
    assert r.returncode == 0, r.stderr
    assert "run solo_kl1 not finished (still running or killed): skipped" in r.stdout
    assert not any("--run-name solo_kl1 " in c + " " or c.endswith("--run-name solo_kl1") for c in calls)
    assert any("solo_kl1_t2048" in c for c in calls)


def test_ablate_no_reference_when_every_window_run_exists(tmp_path):
    r, calls, _ = _ablate(tmp_path, setup={"solo_kl1": "true", "solo_kl1_t2048": "true"})
    assert r.returncode == 0, r.stderr
    assert len(calls) == 2 and all("reeval.sbatch" in c for c in calls)


def test_ablate_refuses_while_pvt_jobs_queued_unless_forced(tmp_path):
    q = tmp_path / "q.sh"
    q.write_bytes(b"#!/bin/bash\necho pvt\n")
    r, calls, _ = _ablate(tmp_path, {"SQUEUE": q.as_posix()})
    assert r.returncode == 1 and "pvt jobs are still queued/running" in r.stderr and calls == []
    (tmp_path / "f").mkdir()
    r, calls, _ = _ablate(tmp_path / "f", {"SQUEUE": q.as_posix(), "FORCE": "1"})
    assert r.returncode == 0 and calls


def _followup(tmp_path, states: dict[str, str], *args, runs="a b c d", reeval="a"):
    out = tmp_path / "out"
    for r in ("a", "b", "c"):
        (out / "trained" / r).mkdir(parents=True, exist_ok=True)
        (out / "trained" / r / "train.json").write_text('{"finished": true}', encoding="utf-8")
    log = tmp_path / "calls.log"
    sb = tmp_path / "sbatch.sh"
    sb.write_bytes(f'#!/bin/bash\necho "$@" >> "{log.as_posix()}"\necho $((RANDOM + 100))\n'.encode())
    sa = tmp_path / "sacct.sh"
    case = "".join(f'  *{j}*) echo "  {s}";;\n' for j, s in states.items())
    sa.write_bytes(f'#!/bin/bash\ncase "$*" in\n{case}  *) echo "";;\nesac\n'.encode())
    env = {**os.environ, "SBATCH": sb.as_posix(), "SACCT": sa.as_posix(), "OUT": out.as_posix(), "RUNS": runs,
           "REEVAL_RUNS": reeval, "VEC_RUNS": "a b"}
    r = subprocess.run([BASH, str(SLURM / "followup.sh"), *args], capture_output=True, text=True, cwd=tmp_path,
                       env=env)
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return r, calls


def test_followup_chains_step1_bench_and_vecs(tmp_path):
    # a: reeval (old eval.json); b: persona after running job 11; c: its job 12 already COMPLETED (no dependency);
    # d: no train.json but its job 13 is running -> after 13
    r, calls = _followup(tmp_path, {"11": "RUNNING", "12": "COMPLETED", "13": "PENDING"}, "b=11", "c=12", "d=13")
    assert r.returncode == 0, r.stderr
    assert len(calls) == 4 * 2 + 1
    assert "reeval.sbatch --out" in calls[0] and "--run-name a" in calls[0] and "--dependency" not in calls[0]
    assert calls[1].startswith("--parsable --dependency=afterok:") and "bench.sbatch" in calls[1]
    assert "--dependency=afterok:11" in calls[2] and "pvt.sbatch persona" in calls[2] and "--run-name b" in calls[2]
    assert "--dependency" not in calls[4] and "--run-name c" in calls[4]
    assert "--dependency=afterok:13" in calls[6] and "--run-name d" in calls[6]
    assert "--dependency" not in calls[8] and "vecs.sbatch" in calls[8] and calls[8].endswith("--runs a b")


def test_followup_skips_a_failed_job_and_a_missing_run(tmp_path):
    r, calls = _followup(tmp_path, {"21": "FAILED"}, "b=21", runs="b d", reeval="")
    assert r.returncode == 0
    assert "b: job 21 ended FAILED; skipped" in r.stderr and "d: no train.json" in r.stderr
    assert len(calls) == 1 and "vecs.sbatch" in calls[0]  # only the vecs job
