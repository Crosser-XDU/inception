#!/usr/bin/env bash
# 逐项试验已有推理优化；模型/权重/数据沿用环境变量或 benchmark_decode_start.sh。
# SPEED_PROFILE=reuse|defer|cache|tmerge|batch|parallel|strict|batch_strict|parallel_strict|target_merge|target_hook
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPEED_PROFILE="${SPEED_PROFILE:-defer}"
export CORRECTION_MODE=defer
export INPLACE_DRAFT_CACHE=0
export MERGE_TARGET_LORA=0
export MERGE_RECURRENT_LORA=0
export ANCHOR_HOOK_HIDDEN_STATES=0
export BATCHED_DRAFT_BOUNDARY=0
export SINGLE_GPU_PARALLEL_DRAFT=0
export FAST_STRICT_VERIFICATION=0
case "$SPEED_PROFILE" in
  reuse) CORRECTION_MODE=reuse ;;
  defer) ;;
  cache) INPLACE_DRAFT_CACHE=1 ;;
  tmerge) INPLACE_DRAFT_CACHE=1; MERGE_RECURRENT_LORA=1 ;;
  batch|parallel|strict|batch_strict|parallel_strict|target_merge|target_hook)
    INPLACE_DRAFT_CACHE=1
    MERGE_RECURRENT_LORA=1
    case "$SPEED_PROFILE" in
      batch|batch_strict|target_merge|target_hook) BATCHED_DRAFT_BOUNDARY=1 ;;
      parallel|parallel_strict) SINGLE_GPU_PARALLEL_DRAFT=1 ;;
    esac
    case "$SPEED_PROFILE" in
      strict|batch_strict|parallel_strict|target_merge|target_hook) FAST_STRICT_VERIFICATION=1 ;;
    esac
    case "$SPEED_PROFILE" in
      target_merge|target_hook) MERGE_TARGET_LORA=1 ;;
    esac
    [[ "$SPEED_PROFILE" != target_hook ]] || ANCHOR_HOOK_HIDDEN_STATES=1
    ;;
  *) printf 'SPEED_PROFILE: reuse / defer / cache / tmerge / batch / parallel / strict / batch_strict / parallel_strict / target_merge / target_hook。\n' >&2; exit 2 ;;
esac
export OUT="${OUT:-$REPO_ROOT/runs/timing_speed_${SPEED_PROFILE}_$(date +%Y%m%d_%H%M%S)_$$}"
printf '优化试验：%s；最终参数以 inputs.json 为准。\n' "$SPEED_PROFILE"
exec bash "$REPO_ROOT/scripts/benchmark_decode_start.sh" "$@"
