#!/usr/bin/env bash
# Linux GPU：填写模型/权重/数据路径后直接运行；CLI 参数优先于环境变量。
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL="${MODEL:-/path/to/Meta-Llama-3-8B-Instruct}"
export CHECKPOINT="${CHECKPOINT:-${CKPT:-/path/to/checkpoint-xxxxx}}"
export DATA="${DATA:-$REPO_ROOT/data/gsm8k_test.jsonl}"
export GPU="${GPU:-0}"
export ROUTE="${ROUTE:-auto}"                     # auto / tail / boundary
export START="${START:-0}"                        # 数据起始行，0 起算
export SAMPLES="${SAMPLES:-32}"
export SPEC_START_POSITION="${SPEC_START_POSITION:-128}" # 已生成多少 token 后开始投机，不含 prompt
export BLOCK="${BLOCK:-3}"                        # 1 枚已确定 token + 最多 BLOCK-1 枚草稿
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
export CORRECTION_MODE="${CORRECTION_MODE:-reuse}"  # 两组保持一致：off / reuse / defer
export RUNTIME_MODE="${RUNTIME_MODE:-throughput}"  # throughput / diagnostic
# 可选运行优化；默认关闭，保留原有对照配置。
export INPLACE_DRAFT_CACHE="${INPLACE_DRAFT_CACHE:-0}"
export MERGE_TARGET_LORA="${MERGE_TARGET_LORA:-0}"
export MERGE_RECURRENT_LORA="${MERGE_RECURRENT_LORA:-0}"
export ANCHOR_HOOK_HIDDEN_STATES="${ANCHOR_HOOK_HIDDEN_STATES:-0}"
export BATCHED_DRAFT_BOUNDARY="${BATCHED_DRAFT_BOUNDARY:-0}"
export SINGLE_GPU_PARALLEL_DRAFT="${SINGLE_GPU_PARALLEL_DRAFT:-0}"
export FAST_STRICT_VERIFICATION="${FAST_STRICT_VERIFICATION:-0}"
export DTYPE="${DTYPE:-bf16}"
export THINKING="${THINKING:-0}"
export REPEATS="${REPEATS:-1}"
export WARMUP_RUNS="${WARMUP_RUNS:-1}"             # 每个进程先预热，不计入比较结果
export DRY_RUN="${DRY_RUN:-0}"
export OUT="${OUT:-$REPO_ROOT/runs/timing_start_$(date +%Y%m%d_%H%M%S)_$$}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="$(command -v python || command -v python3 || true)"
  fi
fi
[[ -n "$PYTHON" ]] || { printf '找不到 Python，请设置 PYTHON。\n' >&2; exit 1; }
cd -- "$REPO_ROOT"
exec "$PYTHON" "$REPO_ROOT/scripts/benchmark_decode_start.py" "$@"
