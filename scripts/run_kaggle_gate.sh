#!/usr/bin/env bash
#
# Stage 0 Kaggle gate automation (T0.0 spike + throughput probe + baseline matrix).
#
# Usage:
#   bash scripts/run_kaggle_gate.sh --stage {setup|spike|probe|matrix|all} [options]
#
# Options:
#   --stage STAGE      setup | spike | probe | matrix | all   (default: all)
#   --models M...      model repo ids or local weight dirs (default: 0.6B + 1.7B)
#   --artifacts DIR    artifact output dir (default: artifacts/stage0)
#   --budget-hours H   per-session budget for the probe (default: 12)
#   --probe-n N        utterances to probe (default: 50)
#   --no-offline       do NOT set HF_HUB_OFFLINE=1 for local --models
#   --dry-run          print the plan; only the matrix runs (with --dry-run)
#   --push             commit AND push the small artifacts (needs git credentials)
#   -h, --help         show this help
#
# Environment: PYTHON (default: python) selects the interpreter.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Ensure `asr1bit` is importable regardless of editable-install behaviour.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STAGE="all"
MODELS=()
MODELS_EXPLICIT=0
ARTIFACTS="${ARTIFACTS:-artifacts/stage0}"
BUDGET_HOURS="12"
PROBE_N="50"
NO_OFFLINE=0
DRY_RUN=0
GATE_PUSH=0
PYTHON="${PYTHON:-python}"

# T4 is sm75: no FlashAttention-2, and vLLM's FlashInfer JIT build fails on it
# (`ninja ... ld returned 1`). Force the self-contained Triton attention backend.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

log() { printf '[gate] %s\n' "$*"; }
die() { printf '[gate] ERROR: %s\n' "$*" >&2; exit 1; }
begin_stage() { log "BEGIN stage=$1 artifacts=$ARTIFACTS"; }

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --stage)
        [[ $# -ge 2 ]] || die "--stage needs a value"
        STAGE="$2"; shift 2 ;;
      --models)
        shift
        MODELS_EXPLICIT=1
        while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
        [[ ${#MODELS[@]} -gt 0 ]] || die "--models needs at least one value" ;;
      --artifacts)
        [[ $# -ge 2 ]] || die "--artifacts needs a value"
        ARTIFACTS="$2"; shift 2 ;;
      --budget-hours)
        [[ $# -ge 2 ]] || die "--budget-hours needs a value"
        BUDGET_HOURS="$2"; shift 2 ;;
      --probe-n)
        [[ $# -ge 2 ]] || die "--probe-n needs a value"
        PROBE_N="$2"; shift 2 ;;
      --no-offline) NO_OFFLINE=1; shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      --push) GATE_PUSH=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown argument: $1 (try --help)" ;;
    esac
  done

  case "$STAGE" in
    setup|spike|probe|matrix|all) ;;
    *) die "invalid --stage '$STAGE' (setup|spike|probe|matrix|all)" ;;
  esac

  if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("Qwen/Qwen3-ASR-0.6B" "Qwen/Qwen3-ASR-1.7B")
  fi
}

record_environment() {
  local phase="$1"
  {
    printf '===== %s %s =====\n' "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$PYTHON" - <<'PY'
import json, platform
report = {"python": platform.python_version()}
for name in ("torch", "transformers", "vllm", "qwen_asr"):
    try:
        module = __import__(name)
        report[name] = getattr(module, "__version__", "unknown")
    except Exception:
        report[name] = None
try:
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        report["device"] = torch.cuda.get_device_name(0)
        report["capability"] = "sm%d%d" % torch.cuda.get_device_capability(0)
except Exception as exc:
    report["torch_error"] = repr(exc)
print(json.dumps(report, indent=2))
PY
  } >>"$ARTIFACTS/environment.txt"
}

stage_preflight() {
  begin_stage preflight
  log "preflight: record versions + assert CUDA"
  mkdir -p "$ARTIFACTS"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] record environment -> $ARTIFACTS/environment.txt, assert CUDA sm70+"
    return 0
  fi
  record_environment "preflight"
  if ! "$PYTHON" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print("torch not importable:", exc)
    sys.exit(1)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False")
    sys.exit(1)
cap = torch.cuda.get_device_capability(0)
if cap[0] < 7:
    print("compute capability sm%d%d < sm70 (vLLM nightly needs sm70+; T4=sm75, P100=sm60)"
          % (cap[0], cap[1]))
    sys.exit(1)
print("CUDA OK: %s sm%d%d" % (torch.cuda.get_device_name(0), cap[0], cap[1]))
PY
  then
    die "CUDA preflight failed (need T4/L4/A100, sm70+)"
  fi
  log "preflight OK"
}

stage_setup() {
  begin_stage setup
  log "setup: vLLM nightly (cu129) + extras + editable install"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] pip install -U vllm --pre (cu129 nightly index)"
    echo '[dry-run] pip install "vllm[audio]" qwen-asr jiwer soundfile soxr pandas pyyaml'
    echo "[dry-run] pip install -e . --no-deps"
    return 0
  fi
  # Kaggle's preinstalled pip is too old for --index-strategy; upgrade best-effort.
  "$PYTHON" -m pip install -q -U pip setuptools wheel || log "WARNING: pip upgrade failed (continuing)"
  # Preferred: let qwen-asr pin a compatible vLLM from PyPI (no special index).
  if ! "$PYTHON" -m pip install -q -U "qwen-asr[vllm]"; then
    log "qwen-asr[vllm] install failed; falling back to vLLM nightly (cu129)"
    "$PYTHON" -m pip install -q -U vllm --pre \
      --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      || die "vLLM install failed (PyPI and nightly)"
  fi
  "$PYTHON" -m pip install -q jiwer soundfile soxr pandas pyyaml \
    || die "runtime extras install failed"
  "$PYTHON" -m pip install -q -e . --no-deps || die "editable install failed"
  if "$PYTHON" -c "import asr1bit, asr1bit.data, asr1bit.eval.harness" 2>/dev/null; then
    log "import check OK (asr1bit)"
  else
    log "import check via editable failed; relying on PYTHONPATH=$REPO_ROOT/src"
  fi
  record_environment "post-setup"
  log "setup OK"
}

stage_spike() {
  begin_stage spike
  log "spike: T0.0 vLLM streaming go/no-go"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] scripts/spike_vllm_streaming.py --model ${MODELS[0]} | tee $ARTIFACTS/spike.log"
    return 0
  fi
  local rc=0
  "$PYTHON" scripts/spike_vllm_streaming.py --model "${MODELS[0]}" 2>&1 \
    | tee "$ARTIFACTS/spike.log" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    {
      echo "[gate] SPIKE FAILED (exit $rc) - streaming gate unreachable on this host."
      echo "[gate] Fallback order:"
      echo "[gate]   1. Colab Pro (A100 40GB / L4 22GB) - same script, GPU runtime."
      echo "[gate]   2. Official Docker image qwenllm/qwen3-asr on a CUDA host."
      echo "[gate] Stopping now: probe and matrix will NOT run."
    } | tee -a "$ARTIFACTS/spike.log" >&2
    exit "$rc"
  fi
  log "spike PASS"
}

stage_probe() {
  begin_stage probe
  log "probe: throughput on clean and other"
  local config
  for config in clean other; do
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] throughput_probe.py --config $config --backend vllm --n $PROBE_N"
      continue
    fi
    "$PYTHON" scripts/throughput_probe.py \
      --backend vllm --n "$PROBE_N" --config "$config" \
      --budget-hours "$BUDGET_HOURS" --results "$ARTIFACTS" \
      || die "throughput probe failed ($config)"
  done
}

stage_matrix() {
  begin_stage matrix
  log "matrix: run_stage0.py --backend vllm (re-runnable; resumes from existing .jsonl)"
  local args=(--backend vllm --results-dir "$ARTIFACTS" --models "${MODELS[@]}")
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi
  if [[ "$MODELS_EXPLICIT" == "1" && "$NO_OFFLINE" != "1" ]]; then
    export HF_HUB_OFFLINE=1
    log "HF_HUB_OFFLINE=1 (ensure LibriSpeech is cached; use --no-offline to override)"
  fi
  "$PYTHON" scripts/run_stage0.py "${args[@]}" || die "matrix failed"
}

collect_artifacts() {
  mkdir -p "$ARTIFACTS"
  log "collecting artifacts -> $ARTIFACTS"
  local manifest="$ARTIFACTS/MANIFEST.txt"
  local names=(environment.txt spike.log reports.csv reports.json)
  {
    echo "# Stage 0 gate manifest"
    echo "stage=$STAGE"
    echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "artifacts_dir=$ARTIFACTS"
    echo
    echo "## files"
    local name
    for name in "${names[@]}"; do
      if [[ -f "$ARTIFACTS/$name" ]]; then
        echo "- $name ($(wc -c <"$ARTIFACTS/$name" | tr -d ' ') bytes)"
      fi
    done
    shopt -s nullglob
    local throughput
    for throughput in "$ARTIFACTS"/throughput_*.json; do
      echo "- $(basename "$throughput") ($(wc -c <"$throughput" | tr -d ' ') bytes)"
    done
    shopt -u nullglob
  } | tee "$manifest"
}

commit_artifacts() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run: skipping artifact commit"
    return 0
  fi
  local name
  for name in environment.txt MANIFEST.txt reports.csv reports.json; do
    if [[ -f "$ARTIFACTS/$name" ]]; then
      git add "$ARTIFACTS/$name"
    fi
  done
  shopt -s nullglob
  local throughput
  for throughput in "$ARTIFACTS"/throughput_*.json; do
    git add "$throughput"
  done
  shopt -u nullglob
  if [[ -f "$ARTIFACTS/spike.log" ]]; then
    git add -f "$ARTIFACTS/spike.log"
  fi
  if git diff --cached --quiet; then
    log "no artifact changes to commit"
    return 0
  fi
  git -c user.name="${GIT_AUTHOR_NAME:-kaggle-gate}" \
      -c user.email="${GIT_AUTHOR_EMAIL:-gate@example.com}" \
      commit -q -m "results(stage0): Kaggle gate artifacts ($STAGE)" \
    || die "artifact commit failed"
  log "committed small artifacts (per-utterance .jsonl left untracked)"

  if [[ "$GATE_PUSH" != "1" ]]; then
    log "set GATE_PUSH=1 to push artifacts"
    return 0
  fi

  local token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [[ -z "$token" ]]; then
    log "WARNING: GATE_PUSH=1 but GITHUB_TOKEN is unset; artifacts committed locally only"
    log "  add a Kaggle Secret named GITHUB_TOKEN (fine-grained, 'contents: write') and export it"
    return 0
  fi
  local remote_url auth_url
  remote_url="$(git remote get-url origin 2>/dev/null || echo "")"
  if [[ -z "$remote_url" ]]; then
    log "WARNING: no 'origin' remote; skipping push"
    return 0
  fi
  auth_url="$(printf '%s' "$remote_url" | sed -E "s#^https://#https://x-access-token:${token}@#")"
  local push_out rc=0
  push_out="$(git push "$auth_url" "HEAD:${GATE_BRANCH:-stage-0}" 2>&1)" || rc=$?
  push_out="${push_out//$token/***}"
  if [[ "$rc" -eq 0 ]]; then
    log "pushed artifacts to origin (${GATE_BRANCH:-stage-0})"
  else
    log "WARNING: push failed (check token scope); artifacts committed locally only"
    printf '%s\n' "$push_out" | sed 's/^/  /'
  fi
  return 0
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  collect_artifacts || true
  exit "$rc"
}

main() {
  parse_args "$@"
  # Always collect artifacts on exit, even on a failed spike or a killed session.
  trap on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  log "stage=$STAGE models=${MODELS[*]} artifacts=$ARTIFACTS dry_run=$DRY_RUN"
  case "$STAGE" in
    setup)  stage_preflight; stage_setup ;;
    spike)  stage_preflight; stage_spike ;;
    probe)  stage_preflight; stage_probe ;;
    matrix) stage_preflight; stage_matrix ;;
    all)    stage_preflight; stage_setup; stage_spike; stage_probe; stage_matrix ;;
  esac
  collect_artifacts
  commit_artifacts
}

main "$@"
