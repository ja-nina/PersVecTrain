# Shared by slurm/run.sbatch and slurm/train.sbatch (sourced, not run). Needs bash.

# pvt_setup_env: python, caches and the vLLM-Hook env vars of PersVecFast's slurm/run_model.sbatch.
pvt_setup_env() {
  PYTHON="${PYTHON:-$(command -v python || true)}"  # active env at sbatch time (sbatch exports PATH)
  [ -x "$PYTHON" ] || { echo "no python found: activate the env (with persvecfast and persvectrain installed) before sbatch" >&2; return 1; }
  export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_USE_FLASHINFER_SAMPLER=0             # its sampler JIT-compiles with nvcc; nodes have no CUDA toolkit
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PWD/.cache/vllm}"  # compile cache off the (small) home quota
  # vLLM-Hook capture files on node-local disk, per job (concurrent jobs must not share them)
  export MIA_APERTURE_DIR="${MIA_APERTURE_DIR:-/tmp/pvt-aperture-${SLURM_JOB_ID:-$$}}"
  export PVF_MIA_DIR="${PVF_MIA_DIR:-/tmp/pvt-mia-${SLURM_JOB_ID:-$$}}"
  mkdir -p "$MIA_APERTURE_DIR"
  trap 'rm -rf "$MIA_APERTURE_DIR" "$PVF_MIA_DIR"' EXIT
}

# pvt_arg_value FLAG ARGS...: prints the value following the first FLAG in ARGS (`--flag value`);
# error (status 1, message on stderr) when FLAG is missing or has no value.
pvt_arg_value() {
  local flag="$1"; shift
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "$flag" ]; then
      [ "$#" -ge 2 ] || { echo "$flag has no value" >&2; return 1; }
      printf '%s' "$2"
      return 0
    fi
    shift
  done
  echo "missing $flag in the arguments (run.sbatch needs --out and --run-name)" >&2
  return 1
}

# pvt_has_flag FLAG ARGS...: status 0 when FLAG is among ARGS.
pvt_has_flag() {
  local flag="$1"; shift
  local a
  for a in "$@"; do [ "$a" = "$flag" ] && return 0; done
  return 1
}
