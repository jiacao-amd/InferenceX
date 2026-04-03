#!/usr/bin/env bash
# =============================================================================
# Evaluate Kimi-K2.5 accuracy using lm-evaluation-harness.
# Starts a vLLM server, runs the eval, then shuts down the server.
#
# Usage (from repo root, inside a ROCm/vLLM container):
#   bash eval_kimi.sh
#
# Key overrideable env vars:
#   MODEL           - path or HF repo (default: local /mnt/models path)
#   MODEL_NAME      - name sent to the OpenAI API (default: same as MODEL)
#   TP              - tensor parallel size (default: 8)
#   EP_SIZE         - expert parallel size (default: 1)
#   PORT            - vLLM server port (default: 8888)
#   MAX_MODEL_LEN   - context length (default: 16384)
#   EVAL_TASK       - lm-eval task yaml (default: gsm8k)
#   NUM_FEWSHOT     - few-shot examples (default: 2)
#   CONC            - concurrent eval requests (default: 32)
#   EVAL_RESULT_DIR - where to write eval JSON results (default: /workspace/eval_results)
# =============================================================================

set -euo pipefail

CURDIR=$(cd "$(dirname "$0")"; pwd)
source "${CURDIR}/benchmarks/benchmark_lib.sh"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve model path: prefer explicit MODEL env var, otherwise auto-detect
# from the HF hub cache under /models (the container mount point).
_DEFAULT_MODEL="moonshotai/Kimi-K2.5"
if [[ -z "${MODEL:-}" ]]; then
    _SNAP_DIR="/models/hub/models--moonshotai--Kimi-K2.5/snapshots"
    if [[ -d "$_SNAP_DIR" ]]; then
        _SNAP=$(ls "$_SNAP_DIR" | head -1)
        if [[ -n "$_SNAP" ]]; then
            _DEFAULT_MODEL="${_SNAP_DIR}/${_SNAP}"
        fi
    fi
fi
export MODEL="${MODEL:-$_DEFAULT_MODEL}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/models/hub}"
export MODEL_NAME="${MODEL_NAME:-$MODEL}"
export TP="${TP:-8}"
export EP_SIZE="${EP_SIZE:-1}"
export PORT="${PORT:-8888}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export EVAL_TASK="${EVAL_TASK:-gsm8k}"
export NUM_FEWSHOT="${NUM_FEWSHOT:-2}"
export CONC="${CONC:-32}"
export EVAL_LIMIT="${EVAL_LIMIT:-}"   # e.g. EVAL_LIMIT=100 for a quick sanity run
# GEN_MAX_TOKENS: max output tokens sent to the API. Must be < MAX_MODEL_LEN.
# Reserve 1024 tokens for the input prompt (2-shot GSM8K ~500-900 tokens).
# Capped at 8192 for the default full run.
_default_gen_out=$(( MAX_MODEL_LEN - 1024 ))
(( _default_gen_out > 8192 )) && _default_gen_out=8192
(( _default_gen_out < 1 )) && _default_gen_out=1
export GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-$_default_gen_out}"
export EVAL_RESULT_DIR="${EVAL_RESULT_DIR:-/workspace/eval_results}"
# Extra args appended verbatim to `vllm serve` (e.g. --compilation-config '...')
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

SERVER_LOG="${SERVER_LOG:-/workspace/server_eval.log}"

echo "======================================================="
echo " Kimi-K2.5 Accuracy Eval"
echo "======================================================="
echo "  MODEL          : $MODEL"
echo "  MODEL_NAME     : $MODEL_NAME"
echo "  TP             : $TP"
echo "  EP_SIZE        : $EP_SIZE"
echo "  PORT           : $PORT"
echo "  MAX_MODEL_LEN  : $MAX_MODEL_LEN"
echo "  EVAL_TASK      : $EVAL_TASK"
echo "  NUM_FEWSHOT    : $NUM_FEWSHOT"
echo "  CONC           : $CONC"
echo "  EVAL_LIMIT     : ${EVAL_LIMIT:-all}"
echo "  GEN_MAX_TOKENS : $GEN_MAX_TOKENS"
echo "  EVAL_RESULT_DIR: $EVAL_RESULT_DIR"
echo "  EXTRA_SERVE_ARGS: ${EXTRA_SERVE_ARGS:-(none)}"
echo "======================================================="
echo ""

mkdir -p "$EVAL_RESULT_DIR"

# ---------------------------------------------------------------------------
# AMD-specific env vars (no-op on NVIDIA)
# ---------------------------------------------------------------------------
if command -v rocm-smi &>/dev/null; then
    version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}' || true)
    if [[ -z "$version" || "$version" -lt 177 ]] 2>/dev/null; then
        export HSA_NO_SCRATCH_RECLAIM=1
    fi
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8
    export VLLM_ROCM_USE_AITER_TRITON_ROPE=1
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    fi
fi

# ---------------------------------------------------------------------------
# Expert parallel flag
# ---------------------------------------------------------------------------
if [ "${EP_SIZE}" -gt 1 ]; then
    EP_FLAG="--enable-expert-parallel"
else
    EP_FLAG=""
fi

# ---------------------------------------------------------------------------
# Start vLLM server
# ---------------------------------------------------------------------------
echo "[Step 1/3] Starting vLLM server..."
# Parse EXTRA_SERVE_ARGS into an array so quoted JSON values are not word-split.
_extra_args=()
if [[ -n "$EXTRA_SERVE_ARGS" ]]; then
    eval "_extra_args=($EXTRA_SERVE_ARGS)"
fi
set -x
vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    ${EP_FLAG} \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_MODEL_LEN" \
    --block-size 1 \
    --trust-remote-code \
    --mm-encoder-tp-mode data \
    "${_extra_args[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

# Ensure server is killed on exit
cleanup() {
    echo ""
    echo "[Cleanup] Stopping vLLM server (PID=$SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "[Cleanup] Server stopped."
}
trap cleanup EXIT

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo ""
echo "[Step 2/3] Running lm-eval (task=${EVAL_TASK}, fewshot=${NUM_FEWSHOT}, conc=${CONC})..."
_limit_arg=()
[[ -n "$EVAL_LIMIT" ]] && _limit_arg=(--limit "$EVAL_LIMIT")

run_lm_eval \
    --port "$PORT" \
    --task "$EVAL_TASK" \
    --num-fewshot "$NUM_FEWSHOT" \
    --results-dir "$EVAL_RESULT_DIR" \
    --concurrent-requests "$CONC" \
    --gen-out-tokens "$GEN_MAX_TOKENS" \
    "${_limit_arg[@]}"

# ---------------------------------------------------------------------------
# Collect and print results
# ---------------------------------------------------------------------------
echo ""
echo "[Step 3/3] Collecting results..."
append_lm_eval_summary

echo ""
echo "======================================================="
echo " Eval complete. Results written to: $(pwd)"
echo "======================================================="

# Print a quick summary of accuracy from any result JSON in the current dir
python3 - <<'PYEOF'
import json, glob, sys

result_files = glob.glob("results_*.json") + glob.glob("*.json")
result_files = [f for f in result_files if f != "meta_env.json"]

parsed = None
for rf in sorted(result_files):
    try:
        with open(rf) as fh:
            data = json.load(fh)
        if "lm_eval_version" in data or "results" in data:
            parsed = data
            print(f"Result file: {rf}")
            break
    except Exception:
        continue

if parsed is None:
    print("No lm-eval result JSON found in current directory.")
    sys.exit(0)

results = parsed.get("results", {})
for task_name, metrics in results.items():
    print(f"\nTask: {task_name}")
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
PYEOF
