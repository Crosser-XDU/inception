#!/usr/bin/env python3
"""Compare greedy, fixed drafting, and fixed drafting after a generated-token position."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

from common import (ROOT, audit_result, checkpoint_metadata, run_logged, sha256, validate_data, write_json)
from run_decode import command, draft_policy_settings


RUNTIME_FLAGS = ("merge_target_lora", "inplace_draft_cache", "merge_recurrent_lora", "anchor_hook_hidden_states",
                 "batched_draft_boundary", "single_gpu_parallel_draft", "fast_strict_verification")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        "SPEC_START_POSITION=128 bash scripts/benchmark_decode_start.sh; "
        "or bash scripts/benchmark_decode_start.sh --spec-start-position 256. "
        "START/--start selects data rows; SPEC_START_POSITION counts generated tokens, excluding prompt."))
    env = os.environ.get
    for name, variable, default in (("model", "MODEL", "/path/to/base"),
            ("checkpoint", "CHECKPOINT", env("CKPT", "/path/to/checkpoint")),
            ("data", "DATA", str(ROOT / "data/gsm8k_test.jsonl")), ("output", "OUT", None)):
        p.add_argument("--" + name, type=Path, default=env(variable, default))
    for name, variable, default in (("start", "START", 0), ("samples", "SAMPLES", 32),
            ("spec-start-position", "SPEC_START_POSITION", 128), ("block", "BLOCK", 3),
            ("max-new-tokens", "MAX_NEW_TOKENS", 1024), ("max-prompt-tokens", "MAX_PROMPT_TOKENS", 1024),
            ("repeats", "REPEATS", 1), ("warmup-runs", "WARMUP_RUNS", 1)):
        p.add_argument("--" + name, type=int, default=env(variable, str(default)))
    p.add_argument("--gpu", default=env("GPU", "0"))
    p.add_argument("--route", choices=["auto", "tail", "boundary"], default=env("ROUTE", "auto"))
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=env("DTYPE", "bf16"))
    p.add_argument("--correction-mode", choices=["off", "reuse", "defer"], default=env("CORRECTION_MODE", "reuse"))
    p.add_argument("--runtime-mode", choices=["diagnostic", "throughput"], default=env("RUNTIME_MODE", "throughput"))
    for name in ("thinking", "dry-run", *(key.replace("_", "-") for key in RUNTIME_FLAGS)):
        value = env(name.upper().replace("-", "_"), "0")
        if value not in {"0", "1"}:
            p.error(name.upper() + " must be 0 or 1.")
        p.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=value == "1")
    args = p.parse_args(argv)
    if min(args.start, args.spec_start_position, args.warmup_runs) < 0:
        p.error("--start, --spec-start-position and --warmup-runs must be non-negative.")
    if min(args.samples, args.repeats, args.max_new_tokens, args.max_prompt_tokens) <= 0 or args.block < 2:
        p.error("Samples/repeats/token limits must be positive; --block must be >= 2.")
    # argparse does not validate string defaults against choices.
    for name, choices in (("route", {"auto", "tail", "boundary"}), ("dtype", {"bf16", "fp16", "fp32"}),
                          ("correction_mode", {"off", "reuse", "defer"}), ("runtime_mode", {"diagnostic", "throughput"})):
        if getattr(args, name) not in choices:
            p.error(f"Invalid {name}: {getattr(args, name)}")
    if args.batched_draft_boundary and args.single_gpu_parallel_draft:
        p.error("Batched boundary and parallel draft are alternative paths; test them separately.")
    if args.inplace_draft_cache and args.correction_mode == "off":
        p.error("This benchmark requires reuse/defer correction when using --inplace-draft-cache.")
    if not args.gpu:
        p.error("A single allocated GPU index or UUID is required.")
    if args.output is None:
        from datetime import datetime
        args.output = ROOT / "runs" / f"timing_start_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    for name in ("model", "checkpoint", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def make_run(args, name, output):
    start_position = 0 if name == "fixed" else args.spec_start_position
    run_args = SimpleNamespace(**(vars(args) | {"output": output, "device": "cuda", "draft_policy": "fixed"}))
    expected = draft_policy_settings(run_args) | {
        "latent_draft_min_position": start_position, "lazy_t_sync": True,
        "defer_t_init_until_latent": True, "no_baseline": name != "fixed", "warmup_runs": args.warmup_runs,
        **{key: getattr(args, key) for key in RUNTIME_FLAGS}}
    cmd = command(run_args) + ["--latent-draft-min-position", str(start_position),
                              "--lazy-t-sync", "--defer-t-init-until-latent", "--decode-order", "alternate", "--warmup-runs", str(args.warmup_runs)]
    if args.fast_strict_verification:
        expected["compact_runtime_stats"] = True
        if "--compact-runtime-stats" not in cmd:
            cmd.append("--compact-runtime-stats")
    cmd.extend("--" + key.replace("_", "-") for key in RUNTIME_FLAGS if getattr(args, key))
    if name != "fixed":
        cmd.append("--no-baseline")
    return cmd, expected


def audit_start_position(data, position):
    first_drafts = []
    for row in data["results"]:
        records = row["adaptive"].get("block_records")
        if not isinstance(records, list):
            raise ValueError("Missing block_records for start-position verification.")
        drafts = [r for r in records if r["proposed_len"] > 1]
        if any(r["generated_start"] < position for r in drafts):
            raise ValueError("Drafting occurred before the requested start position.")
        if sum(r["proposed_len"] - 1 for r in drafts) != row["adaptive"]["draft_tokens"]:
            raise ValueError("Draft records disagree with draft_tokens.")
        first_drafts.append({"sample": row["sample"], "first_draft_position":
                             min(r["generated_start"] + 1 for r in drafts) if drafts else None})
    return {"spec_start_position": position, "drafting_samples": sum(r["first_draft_position"] is not None
            for r in first_drafts), "first_drafts": first_drafts}


def audit_runtime_optimizations(data):
    keys = ("batched_boundary_blocks", "parallel_draft_steps", "fast_strict_blocks", "fast_strict_draft_tokens")
    args = data["summary"]["args"]
    enabled = any(args.get(key) for key in ("batched_draft_boundary", "single_gpu_parallel_draft", "fast_strict_verification"))
    totals = data["summary"].get("runtime_optimization_counts")
    if totals is None and not enabled:
        return {}  # Old results remain readable when no new optimization was requested.
    if not isinstance(totals, dict):
        raise ValueError("Missing optimization execution counters; sync the updated decoder.")
    for key in keys:
        counts = [totals.get(key), *(r["adaptive"].get(key) for r in data["results"])]
        if any(type(n) is not int or n < 0 for n in counts) or counts[0] != sum(counts[1:]):
            raise ValueError(f"Missing or inconsistent runtime counter: {key}")
    for row in data["results"]:
        r = row["adaptive"]
        blocks = r["block_records"]
        drafts = sum(b["proposed_len"] > 1 for b in blocks)
        if r["batched_boundary_blocks"] != (drafts if args.get("batched_draft_boundary") else 0):
            raise ValueError("Batched boundary did not execute on the requested draft blocks.")
        if not args.get("single_gpu_parallel_draft") and r["parallel_draft_steps"]:
            raise ValueError("Unrequested parallel drafting executed.")
        if args.get("single_gpu_parallel_draft") and any(b["proposed_len"] >= 3 for b in blocks) and not r["parallel_draft_steps"]:
            raise ValueError("Parallel drafting was requested but no overlapping steps executed.")
        if r["fast_strict_blocks"] != (len(blocks) if args.get("fast_strict_verification") else 0):
            raise ValueError("Fast strict verification did not cover all verified blocks.")
        if r["fast_strict_draft_tokens"] != (r["draft_tokens"] if args.get("fast_strict_verification") else 0):
            raise ValueError("Fast strict verification draft coverage disagrees with generated drafts.")
    return totals


def comparison(fixed, delayed, position):
    left, right = fixed["results"], delayed["results"]
    if not left or len(left) != len(right):
        raise ValueError("Run coverage differs.")
    shared_flags = ("model_name_or_path", "checkpoint", "data_file", "start_index", "max_samples",
        "max_new_tokens", "max_prompt_tokens", "max_block_tokens", "dtype", "draft_logit_source",
        "question_prefix", "question_suffix", "enable_thinking", "disable_thinking", "mode",
        "cheap_verifier_policy", "lazy_t_sync", "defer_t_init_until_latent", "compact_runtime_stats",
        "production_async_timing", "reuse_verify_cache_for_correction", "defer_correction_to_next_verify", "warmup_runs", *RUNTIME_FLAGS)
    if any(fixed["summary"]["args"].get(k) != delayed["summary"]["args"].get(k) for k in shared_flags):
        raise ValueError("The two runs differ in shared experiment settings.")
    for f, d in zip(left, right):
        if any(f[k] != d[k] for k in ("sample", "question", "reference_answer", "prompt_tokens", "answer_metric")):
            raise ValueError("Runs differ in sample identity, order, reference, or prompt length.")
        if f["baseline"] is None or d["baseline"] is not None:
            raise ValueError("Expected one shared greedy baseline in the fixed run only.")
    metrics = []
    baseline_ids = [row["baseline"]["token_ids"] for row in left]
    for name, rows, key, correct_key in (("greedy", left, "baseline", "baseline_correct"),
            ("fixed", left, "adaptive", "correct"), ("after_start", right, "adaptive", "correct")):
        records = [r[key] for r in rows]
        seconds = sum(r["wall_time_s"] for r in records)
        tokens = sum(len(r["token_ids"]) for r in records)
        if seconds <= 0 or tokens <= 0:
            raise ValueError("Empty generation or non-positive wall time cannot be benchmarked.")
        drafts = sum(r.get("draft_tokens", 0) for r in records)
        accepted = sum(r.get("accepted_draft_tokens", 0) for r in records)
        draft_blocks = [b for r in records for b in r.get("block_records", []) if b["proposed_len"] > 1]
        metrics.append({"method": name, "samples": len(rows), "time_s": seconds, "tokens": tokens,
            "tokens_per_s": tokens / seconds, "correct": sum(r[correct_key] is True for r in rows),
            "accuracy": sum(r[correct_key] is True for r in rows) / len(rows),
            "exact_matches": sum(r["token_ids"] == b for r, b in zip(records, baseline_ids)),
            "exact_rate": sum(r["token_ids"] == b for r, b in zip(records, baseline_ids)) / len(rows),
            "draft_tokens": drafts, "accepted_draft_tokens": accepted,
            **{key: sum(r.get(key, 0) for r in records) for key in
               ("batched_boundary_blocks", "parallel_draft_steps", "fast_strict_blocks", "fast_strict_draft_tokens")},
            "draft_blocks": len(draft_blocks),
            "zero_accept_blocks": sum(b["accepted_len"] == 1 for b in draft_blocks),
            "full_accept_blocks": sum(b["accepted_len"] == b["proposed_len"] for b in draft_blocks),
            "target_calls": sum(r.get("target_calls", 0) for r in records),
            "correction_cache_reuses": sum(r.get("fast_correction_cache_reuses", 0) for r in records),
            "deferred_corrections": sum(r.get("deferred_corrective_tokens", 0) for r in records),
            "draft_acceptance": accepted / drafts if drafts else None,
            "drafting_samples": sum(r.get("draft_tokens", 0) > 0 for r in records),
            "spec_start_position": position if name == "after_start" else 0})
    for m in metrics:
        add_execution_rates(m)
        m["wall_speedup"] = metrics[0]["time_s"] / m["time_s"]
        m["throughput_speedup"] = m["tokens_per_s"] / metrics[0]["tokens_per_s"]
    return metrics


def add_execution_rates(row):
    blocks = row["draft_blocks"]
    row["mean_accepted_drafts"] = row["accepted_draft_tokens"] / blocks if blocks else None
    row["zero_accept_rate"] = row["zero_accept_blocks"] / blocks if blocks else None
    row["full_accept_rate"] = row["full_accept_blocks"] / blocks if blocks else None


def aggregate(repeats):
    rows = []
    for i in range(3):
        parts = [repeat["metrics"][i] for repeat in repeats]
        row = {k: sum(p[k] for p in parts) for k in ("samples", "time_s", "tokens", "correct", "exact_matches",
                    "draft_tokens", "accepted_draft_tokens", "drafting_samples", "draft_blocks", "zero_accept_blocks",
                    "full_accept_blocks", "target_calls", "correction_cache_reuses", "deferred_corrections",
                    "batched_boundary_blocks", "parallel_draft_steps", "fast_strict_blocks", "fast_strict_draft_tokens")}
        row.update(method=parts[0]["method"], spec_start_position=parts[0]["spec_start_position"],
            tokens_per_s=row["tokens"] / row["time_s"], accuracy=row["correct"] / row["samples"],
            exact_rate=row["exact_matches"] / row["samples"],
            draft_acceptance=row["accepted_draft_tokens"] / row["draft_tokens"] if row["draft_tokens"] else None)
        add_execution_rates(row)
        rows.append(row)
    for row in rows:
        row["wall_speedup"] = rows[0]["time_s"] / row["time_s"]
        row["throughput_speedup"] = row["tokens_per_s"] / rows[0]["tokens_per_s"]
    return rows


def save_report(args, repeats):
    rows = aggregate(repeats)
    write_json(args.output / "comparison.json", {"spec_start_position": args.spec_start_position,
        "data_start": args.start, "unique_samples": args.samples, "repeats": repeats, "aggregate": rows,
        "baseline_source": "Each repeat's fixed/result.json results[].baseline; shared by both comparisons."})
    with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"同一 checkpoint 三组对照：{args.checkpoint}",
        f"数据区间 [{args.start}, {args.start + args.samples})；重复 {len(repeats)} 次；投机起始位置 {args.spec_start_position}",
        f"BLOCK={args.block}（最多 {args.block - 1} 枚草稿）；route={args.route}；correction={args.correction_mode}；runtime={args.runtime_mode}",
        "运行优化：" + ", ".join(f"{key}={int(getattr(args, key))}" for key in RUNTIME_FLAGS),
        "方法           总秒数     tokens/s   耗时加速比  吞吐加速比  准确率   完整输出一致率"]
    for row in rows:
        lines.append(f"{row['method']:12s} {row['time_s']:10.3f} {row['tokens_per_s']:10.3f} "
            f"{row['wall_speedup']:10.3f} {row['throughput_speedup']:10.3f} "
            f"{row['accuracy']:8.2%} {row['exact_rate']:10.2%}")
        if row["method"] != "greedy":
            lines.append(f"  tokens={row['tokens']}；接受/草稿={row['accepted_draft_tokens']}/{row['draft_tokens']}；"
                         f"触发投机的样本运行数={row['drafting_samples']}/{row['samples']}")
            if row["draft_blocks"]:
                lines.append(f"  草稿轮数={row['draft_blocks']}；每草稿轮平均接受={row['mean_accepted_drafts']:.3f}；"
                             f"零接受比例={row['zero_accept_rate']:.2%}；全部接受比例={row['full_accept_rate']:.2%}")
            lines.append(f"  目标调用={row['target_calls']}；纠正缓存复用={row['correction_cache_reuses']}；"
                         f"延迟纠正={row['deferred_corrections']}")
            lines.append(f"  批量 boundary 块={row['batched_boundary_blocks']}；并行草稿步={row['parallel_draft_steps']}；"
                         f"轻量严格验证块={row['fast_strict_blocks']}")
    lines += [f"计时包含 prefill、T 初始化和生成，不含模型加载及每个进程的 {args.warmup_runs} 次预热。吞吐模式的分项不是 GPU 耗时。",
        "两组投机均启用延迟 T 初始化和批量缓存同步；起始位置前无草稿，仍有现有目标/缓存维护开销。",
        "每个 repeat 只测一次 greedy，两组使用同一份基线；重复时交替运行 fixed/after_start 的先后顺序。",
        "GPU 观察状态：" + "; ".join(f"repeat{r['repeat']} {r['timing_status']}" for r in repeats)]
    if rows[2]["drafting_samples"] == 0:
        lines.append("after_start 未触发投机：回答可能提前结束，或生成预算未覆盖起始位置。")
    report = "\n".join(lines) + "\n"
    (args.output / "comparison.txt").write_text(report, encoding="utf-8")
    print(report)


def main(argv=None):
    args = parse_args(argv)
    if not (args.model / "config.json").is_file():
        raise ValueError("MODEL/--model must contain config.json.")
    metadata = checkpoint_metadata(args.checkpoint, "tail")
    if args.route == "auto":
        args.route = "boundary" if metadata.get("boundary_head_rank", 0) > 0 else "tail"
    checkpoint_metadata(args.checkpoint, args.route)
    if args.batched_draft_boundary and (args.route != "boundary" or metadata.get("token_conditioning_rank", 0) > 0):
        raise ValueError("Batched boundary requires boundary route and a checkpoint without token conditioning.")
    coverage = validate_data(args.data, args.start, args.samples)
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    def fingerprints():
        return {str(path): sha256(path) for path in [args.data, *(args.checkpoint / name for name in
            ("adapter_config.json", "adapter_model.safetensors", "recurft_config.json", "recurft_recurrent.safetensors",
             "tokenizer_config.json", "tokenizer.json"))]}
    assets = fingerprints()
    if args.dry_run:
        for name in ("fixed", "after_start"):
            cmd, _ = make_run(args, name, args.output / "repeat1" / name)
            print(f"{name}: {shlex.join(cmd)}", flush=True)
        print("预检通过：未运行模型、未检查实时 GPU、未创建输出目录。")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "inputs.json", {"arguments": {k: str(v) if isinstance(v, Path) else v
               for k, v in vars(args).items()}, "data": coverage, "checkpoint": metadata, "asset_sha256": assets})
    repeats = []
    try:
        for repeat in range(1, args.repeats + 1):
            runs, statuses = {}, {}
            order = ("fixed", "after_start") if repeat % 2 else ("after_start", "fixed")
            for name in order:
                if fingerprints() != assets:
                    raise ValueError("Data or checkpoint changed during the comparison.")
                output = args.output / f"repeat{repeat}" / name
                output.mkdir(parents=True, exist_ok=False)
                cmd, expected = make_run(args, name, output)
                print(f"repeat{repeat}/{name}: {shlex.join(cmd)}\nLog: {output / 'run.log'}", flush=True)
                gpu = run_logged(cmd, output, args.gpu)
                audit = audit_result(output / "result.json", args.route, args.start, args.samples,
                                     correction_mode=args.correction_mode, draft_policy=expected)
                data = json.loads((output / "result.json").read_text(encoding="utf-8"))
                if args.merge_target_lora and data["summary"].get("target_lora_merged") is not True:
                    raise ValueError("Target LoRA merge was requested but not reported as applied.")
                if args.merge_recurrent_lora and (not data["summary"].get("recurrent_lora_merged")
                        or data["summary"].get("merged_recurrent_lora_modules", 0) <= 0):
                    raise ValueError("T LoRA merge was requested but no merged modules were reported.")
                audit["runtime_optimization_counts"] = audit_runtime_optimizations(data)
                audit.update(audit_start_position(data, expected["latent_draft_min_position"]))
                audit["timing_status"] = gpu["timing_status"]
                if expected["no_baseline"]:
                    audit.update(baseline_correct=None, wrong_to_right=None, right_to_wrong=None,
                                 baseline_source="../fixed/result.json; metrics computed in comparison.json")
                write_json(output / "complete.marker.json", audit)
                runs[name], statuses[name] = data, gpu["timing_status"]
            if fingerprints() != assets:
                raise ValueError("Data or checkpoint changed during the comparison.")
            repeats.append({"repeat": repeat, "order": list(order), "timing_status": statuses,
                "metrics": comparison(runs["fixed"], runs["after_start"], args.spec_start_position)})
            save_report(args, repeats)
        write_json(args.output / "complete.marker.json", {"completed_repeats": len(repeats), "asset_sha256": assets})
        print(f"结果已保存：{args.output / 'comparison.txt'}")
    except Exception as exc:
        write_json(args.output / "failed.marker.json", {"error": str(exc), "completed_repeats": len(repeats)})
        raise


if __name__ == "__main__":
    main()
