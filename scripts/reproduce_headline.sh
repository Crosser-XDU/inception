#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
: "${MODEL:?Set MODEL to the local Llama-3-8B-Instruct base directory}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained boundary checkpoint-60375}"
: "${DATA:?Set DATA to the full 1319-row GSM8K test JSONL}"
: "${OUT:?Set OUT to a new output directory}"
: "${GPU:?Set GPU to a physical GPU index or UUID}"
if [[ $# -ne 0 ]]; then
  printf 'Use environment variables; arbitrary argument overrides are disabled.\n' >&2
  exit 2
fi
EXTRA=()
if [[ ${DRY_RUN:-0} == 1 ]]; then EXTRA+=(--dry-run); fi
for repeat in 1 2; do
  "$PYTHON" "$ROOT/scripts/run_decode.py" \
    --model "$MODEL" --checkpoint "$CHECKPOINT" --data "$DATA" \
    --output "$OUT/repeat$repeat" --gpu "$GPU" --route boundary \
    --start 8 --samples 384 --max-new-tokens 256 --max-prompt-tokens 1024 \
    --block 3 --dtype fp16 "${EXTRA[@]}"
done
