#!/usr/bin/env bash
# 修改下面的参数，在 Linux GPU 节点运行：bash scripts/benchmark_decode_cache.sh
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 同名环境变量优先；已有训练权重即可，无需重新训练。
export MODEL="${MODEL:-/path/to/Qwen3-8B}"
export CHECKPOINT="${CHECKPOINT:-${CKPT:-/path/to/checkpoint-xxxxx}}"
export DATA="${DATA:-$REPO_ROOT/data/gsm8k_test.jsonl}"
export GPU="${GPU:-0}"
export ROUTE="${ROUTE:-auto}"                  # auto / tail / boundary
export CORRECTION_MODE="${CORRECTION_MODE:-reuse}"  # reuse / defer / off
export SAMPLES="${SAMPLES:-32}"
export START="${START:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
export BLOCK="${BLOCK:-3}"
export DTYPE="${DTYPE:-bf16}"
export REPEATS="${REPEATS:-1}"
export THINKING="${THINKING:-0}"
export DRY_RUN="${DRY_RUN:-0}"
export PYTHON="${PYTHON:-}"
export OUT="${OUT:-$REPO_ROOT/runs/timing_cache_${CORRECTION_MODE}_$(date +%Y%m%d_%H%M%S)_$$}"

for argument in "$@"; do
  case "$argument" in
    -h|--help)
      cat <<'HELP'
缓存优化测速：先填写 MODEL、CHECKPOINT，确认 DATA 和 GPU。
  bash scripts/benchmark_decode_cache.sh --dry-run
  bash scripts/benchmark_decode_cache.sh

默认 CORRECTION_MODE=reuse：保留验证 KV，只计算纠正 token。
  CORRECTION_MODE=defer bash scripts/benchmark_decode_cache.sh
    将纠正 token 合并到下一轮验证，避免独立的纠正前向。
  CORRECTION_MODE=off bash scripts/benchmark_decode_cache.sh
    使用原纠正路径，作为同一 checkpoint 的对照。

每种模式单独运行，不同时叠加 reuse 和 defer 开关。
沿用原脚本的严格 target-match、n-gram off、GPU 检查与同步分项计时。
每遍同时跑 greedy 基线；结果含模式、缓存复用/延迟纠正次数、速度与质量。
--dry-run 不启动模型、不检查实时 GPU、不创建输出目录。
HELP
      exit 0 ;;
  esac
done

exec bash "$REPO_ROOT/scripts/benchmark_decode.sh" "$@"
