#!/usr/bin/env bash
# Linux GPU 节点：bash scripts/benchmark_decode.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== 常改参数：修改默认值即可；同名环境变量优先 =====
MODEL="${MODEL:-/path/to/Meta-Llama-3-8B-Instruct}"
CHECKPOINT="${CHECKPOINT:-${CKPT:-/path/to/checkpoint-xxxxx}}"
DATA="${DATA:-$REPO_ROOT/data/gsm8k_test.jsonl}"  # question/answer JSONL
GPU="${GPU:-}"                                # 必填：分配卡的 UUID 或物理编号
ROUTE="${ROUTE:-auto}"                         # auto / tail / boundary
SAMPLES="${SAMPLES:-32}"                       # 题数，不是训练步数
START="${START:-0}"                           # 题目起始行，0 起算
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
BLOCK="${BLOCK:-3}"                           # 已确定 token + 最多 BLOCK-1 枚草稿
DTYPE="${DTYPE:-bf16}"                         # bf16 / fp16 / fp32
REPEATS="${REPEATS:-1}"                        # 同一批题目重复几遍
THINKING="${THINKING:-0}"                      # 0 / 1
DRY_RUN="${DRY_RUN:-0}"                        # 1 只预检
PYTHON="${PYTHON:-}"                           # 留空优先使用 .venv/bin/python
OUT="${OUT:-}"                                # 留空自动生成；须是新目录
# ==================================================

usage() {
  cat <<'HELP'
修改脚本开头的参数，然后运行：
  bash scripts/benchmark_decode.sh --dry-run
  bash scripts/benchmark_decode.sh

也可以用环境变量覆盖参数：
  MODEL=/path/to/base CHECKPOINT=/path/to/checkpoint GPU=GPU-xxx \
    SAMPLES=128 REPEATS=2 bash scripts/benchmark_decode.sh

auto 根据 boundary_head_rank 选路；头部配置存在不等于已经充分训练。
core / multistep 权重用 tail，完成 boundary 训练后可用 boundary。
需要已有模型、checkpoint 和评测 JSONL；本脚本不下载资产。
每遍比较同一 checkpoint 的 greedy 与投机解码。
输出：OUT/repeatN/{run.log,result.json,timing_summary.txt,complete.marker.json,...}
相对路径以项目根目录为基准。诊断计时包含 prefill 和生成，不含模型加载。
HELP
}
fail() { printf '错误：%s\n' "$*" >&2; exit 1; }

for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "未知参数 $argument；请修改顶部参数或使用环境变量。" ;;
  esac
done

cd -- "$REPO_ROOT"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="$(command -v python || command -v python3 || true)"
  fi
fi
[[ -n "$PYTHON" ]] && command -v "$PYTHON" >/dev/null 2>&1 \
  || fail '找不到 Python，请设置 PYTHON 为训练环境的 Python 路径。'
[[ -n "$GPU" ]] || fail '请设置 GPU 为集群分配给你的空闲卡 UUID 或物理编号。'
[[ -f "$MODEL/config.json" ]] || fail "MODEL 不是完整基座模型目录：$MODEL"
[[ -d "$CHECKPOINT" ]] || fail "找不到 checkpoint 目录：$CHECKPOINT"
[[ -f "$DATA" ]] || fail "找不到评测 JSONL：$DATA；请准备含 question 和 answer 的数据文件。"
case "$ROUTE" in auto|tail|boundary) ;; *) fail 'ROUTE 必须是 auto、tail 或 boundary。' ;; esac
case "$DTYPE" in bf16|fp16|fp32) ;; *) fail 'DTYPE 必须是 bf16、fp16 或 fp32。' ;; esac
for name in SAMPLES MAX_NEW_TOKENS MAX_PROMPT_TOKENS BLOCK REPEATS; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name 必须是正整数。"
done
[[ "$START" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'START 必须是非负整数。'
[[ "$BLOCK" -ge 2 ]] || fail 'BLOCK 至少为 2，才能尝试生成草稿。'
for name in THINKING DRY_RUN; do
  [[ "${!name}" == 0 || "${!name}" == 1 ]] || fail "$name 必须是 0 或 1。"
done

ROUTE="$("$PYTHON" - "$CHECKPOINT" "$ROUTE" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]).resolve()
metadata_path = p / "recurft_config.json"
if not metadata_path.is_file():
    raise SystemExit(f"缺少递归配置：{metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
rank = metadata.get("boundary_head_rank", 0)
route = sys.argv[2]
if route == "auto":
    route = "boundary" if rank > 0 else "tail"
if route == "boundary" and rank <= 0:
    raise SystemExit("该 checkpoint 没有 boundary 头，请使用 ROUTE=tail。")
state_path = p / "trainer_state.json"
step = "未知（目录中没有 trainer_state.json）"
if state_path.is_file():
    step = json.loads(state_path.read_text(encoding="utf-8")).get("global_step", "未知")
print(f"Checkpoint: {p}\n累计训练 global_step: {step}\nboundary_head_rank: {rank}", file=sys.stderr)
print(route)
PY
)"

OUT="${OUT:-$REPO_ROOT/runs/timing_${ROUTE}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" ]] || fail "输出目录已经存在，请更换 OUT：$OUT"
printf 'Python: %s\nGPU: %s\nRoute: %s\nSamples: %s\n输出目录: %s\n' \
  "$PYTHON" "$GPU" "$ROUTE" "$SAMPLES" "$OUT"

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  run_dir="$OUT/repeat$repeat"
  cmd=("$PYTHON" "$REPO_ROOT/scripts/run_decode.py"
    --model "$MODEL" --checkpoint "$CHECKPOINT" --data "$DATA"
    --output "$run_dir" --gpu "$GPU" --route "$ROUTE"
    --samples "$SAMPLES" --start "$START"
    --max-new-tokens "$MAX_NEW_TOKENS" --max-prompt-tokens "$MAX_PROMPT_TOKENS"
    --block "$BLOCK" --dtype "$DTYPE")
  if [[ "$THINKING" == 1 ]]; then cmd+=(--thinking); fi
  if [[ "$DRY_RUN" == 1 ]]; then cmd+=(--dry-run); fi
  printf '\n第 %s/%s 次；日志：%s/run.log\n' "$repeat" "$REPEATS" "$run_dir"
  if "${cmd[@]}"; then
    if [[ "$DRY_RUN" == 1 ]]; then continue; fi
  else
    status=$?
    printf '运行未完成，退出码 %s。若日志已生成，请查看：%s/run.log\n' "$status" "$run_dir" >&2
    exit "$status"
  fi

  "$PYTHON" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
s = json.loads((run / "result.json").read_text(encoding="utf-8"))["summary"]
marker = json.loads((run / "complete.marker.json").read_text(encoding="utf-8"))
fields = [
    ("checkpoint", "Checkpoint"),
    ("samples", "题数"),
    ("baseline_time_s", "Greedy 总耗时（秒）"),
    ("adaptive_time_s", "投机总耗时（秒）"),
    ("wall_clock_speedup", "耗时加速比（大于 1 才加速）"),
    ("baseline_tokens_per_s", "Greedy tokens/s"),
    ("adaptive_tokens_per_s", "投机 tokens/s"),
    ("token_throughput_speedup", "Token 吞吐加速比"),
    ("baseline_generated_tokens", "Greedy 生成 token 数"),
    ("adaptive_generated_tokens", "投机生成 token 数"),
    ("draft_tokens", "草稿 token 数"),
    ("accepted_draft_tokens", "接受草稿 token 数"),
    ("draft_acceptance", "草稿接受率（0～1）"),
    ("target_call_speedup", "目标模型调用次数加速比"),
    ("exact_baseline_rate", "完整输出一致率（0～1）"),
    ("baseline_accuracy", "Greedy 答题准确率（0～1）"),
    ("accuracy", "投机答题准确率（0～1）"),
    ("component_timing_mode", "分项计时方式"),
]
lines = ["测速结果"]
for key, label in fields:
    value = s.get(key)
    value = f"{value:.6f}" if isinstance(value, float) else str(value)
    lines.append(f"{label}: {value}")
lines.append(f"GPU 观察状态: {marker.get('timing_status', '未知')}")
if s.get("component_timing_mode") == "host_enqueue":
    lines.append("分项是主机提交耗时，不能据此判断 GPU 各环节占比。")
lines.append("\n分项累计秒数（降序；部分统计范围可能重叠，不直接求和）：")
for key, value in sorted(s["adaptive_timing_sums"].items(), key=lambda item: item[1], reverse=True):
    lines.append(f"{key:32s} {value:.6f}")
report = "\n".join(lines) + "\n"
(run / "timing_summary.txt").write_text(report, encoding="utf-8")
print(report)
print(f"简报已保存：{run / 'timing_summary.txt'}")
PY
done

if [[ "$DRY_RUN" == 1 ]]; then
  printf '\n预检通过；尚未检查实时 GPU 状态，也未执行推理或创建输出目录。\n'
else
  printf '\n全部完成：%s\n' "$OUT"
fi
