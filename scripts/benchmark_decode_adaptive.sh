#!/usr/bin/env bash
# Linux GPU 节点：填写下面参数后运行 bash scripts/benchmark_decode_adaptive.sh
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 同名环境变量优先。使用已经训练好的 checkpoint，无需重新训练。
export MODEL="${MODEL:-/path/to/Meta-Llama-3-8B-Instruct}"
export CHECKPOINT="${CHECKPOINT:-${CKPT:-/path/to/checkpoint-xxxxx}}"
export DATA="${DATA:-$REPO_ROOT/data/gsm8k_test.jsonl}"
export GPU="${GPU:-0}"
export ROUTE="${ROUTE:-auto}"                    # core / multistep 可显式选 tail
export DRAFT_POLICY="${DRAFT_POLICY:-adaptive}"  # adaptive / schedule（只按位置）/ fixed（固定长度）
export CORRECTION_MODE="${CORRECTION_MODE:-reuse}"  # reuse / defer / off；对照时保持一致

# 原始 logit 的 top-1/top-2 差值，不是概率。以下阈值尚未在你的 checkpoint 上校准。
export TARGET_SKIP_MARGIN="${TARGET_SKIP_MARGIN:-1.0}"   # 小于此值：不尝试草稿；0 关闭
export TARGET_SHORT_MARGIN="${TARGET_SHORT_MARGIN:-3.0}" # 小于此值：最多 1 枚草稿；0 关闭
export DRAFT_MIN_MARGIN="${DRAFT_MIN_MARGIN:-1.0}"       # 上枚草稿差值不足：停止继续展开；0 关闭
export COOLDOWN_FAILURES="${COOLDOWN_FAILURES:-3}"       # 连续 3 个草稿块零接受，触发冷却
export COOLDOWN_CYCLES="${COOLDOWN_CYCLES:-4}"           # 暂停 4 轮再试；两项同时 0 可关闭
export BLOCK="${BLOCK:-3}"                              # 总块上限 = 1 枚已确定 token + 草稿；5 表示最多 4 枚草稿

# 未设置时各段沿用 BLOCK；位置从已生成的第 0 个 token 起算，不含 prompt。
export SHORT_BLOCK="${SHORT_BLOCK:-$BLOCK}"
export MEDIUM_BLOCK="${MEDIUM_BLOCK:-$BLOCK}"
export LONG_BLOCK="${LONG_BLOCK:-$BLOCK}"
export MEDIUM_POSITION="${MEDIUM_POSITION:-96}"
export LONG_POSITION="${LONG_POSITION:-192}"
export RUNTIME_MODE="${RUNTIME_MODE:-diagnostic}"  # throughput 减少诊断，完整 wall 计时仍同步

export SAMPLES="${SAMPLES:-32}"
export START="${START:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
export DTYPE="${DTYPE:-bf16}"
export REPEATS="${REPEATS:-1}"
export THINKING="${THINKING:-0}"
export DRY_RUN="${DRY_RUN:-0}"
export PYTHON="${PYTHON:-}"
export OUT="${OUT:-$REPO_ROOT/runs/timing_${DRAFT_POLICY}_${CORRECTION_MODE}_$(date +%Y%m%d_%H%M%S)_$$}"

for argument in "$@"; do
  case "$argument" in
    -h|--help)
      cat <<'HELP'
自适应投机解码测速：填写 MODEL、CHECKPOINT，确认 DATA 和分配的 GPU。
  bash scripts/benchmark_decode_adaptive.sh --dry-run
  bash scripts/benchmark_decode_adaptive.sh

默认 adaptive + reuse，每遍同时测同一 checkpoint 的 greedy 基线。
目标 top-1/top-2 原始 logit 差值 m：
  m < TARGET_SKIP_MARGIN：不尝试草稿。
  否则 m < TARGET_SHORT_MARGIN：最多尝试 1 枚草稿。
  否则：最多 BLOCK-1 枚草稿，每步由 DRAFT_MIN_MARGIN 决定是否继续。
低置信度草稿仍交给目标模型验证，只停止生成后续草稿。
连续 COOLDOWN_FAILURES 个草稿块零接受，暂停 COOLDOWN_CYCLES 轮再试。
所有草稿保留严格 target-match 验证；n-gram 关闭。

修改上限或运行固定长度对照（其他参数保持相同）：
  BLOCK=5 bash scripts/benchmark_decode_adaptive.sh
  DRAFT_POLICY=fixed bash scripts/benchmark_decode_adaptive.sh
  CORRECTION_MODE=defer bash scripts/benchmark_decode_adaptive.sh
按位置提高草稿上限：前 128 token 最多 1 枚，128～511 最多 2 枚，512 起最多 4 枚：
  BLOCK=5 SHORT_BLOCK=2 MEDIUM_BLOCK=3 LONG_BLOCK=5 \
    MEDIUM_POSITION=128 LONG_POSITION=512 MAX_NEW_TOKENS=2048 \
    bash scripts/benchmark_decode_adaptive.sh
DRAFT_POLICY=schedule 只按位置控制长度，不使用置信度门控和冷却。
RUNTIME_MODE=throughput 减少详细诊断和逐组件 CUDA 同步，greedy/投机整段计时仍同步。
throughput 分项计时是主机提交时间，不能作为 GPU 耗时分解；与对照保持相同模式。
阈值设 0 关闭相应判断；冷却的两项需同时设 0。
TARGET_SHORT_MARGIN 启用时应 >= TARGET_SKIP_MARGIN。
这些默认阈值未经 checkpoint 校准，不保证提速。

输出 OUT/repeatN/{run.log,result.json,timing_summary.txt,complete.marker.json,...}。
统计速度、质量、门控/冷却次数、无草稿比例、平均接受长度及块长度分布。
无草稿路径仍沿用现有缓存维护；开关不等于完整的 greedy 快速路径。
固定长度对照会忽略门控阈值和冷却设置，并要求各段上限等于 BLOCK。
依赖同版本 benchmark_decode.sh、run_decode.py、common.py，以及已有解码器。
--dry-run 只做文件/参数预检，不启动模型、不创建输出目录。
HELP
      exit 0 ;;
  esac
done

exec bash "$REPO_ROOT/scripts/benchmark_decode.sh" "$@"
