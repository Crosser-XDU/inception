#!/usr/bin/env python3
"""Run one explicit, strict proposal route and verify what actually executed."""

import argparse
import json
import math
from pathlib import Path
import shlex
import sys

from common import EXPERIMENT, audit_result, checkpoint_metadata, run_logged, validate_data, write_json


def draft_policy_settings(args):
    policy = getattr(args, "draft_policy", "fixed")
    adaptive = policy == "adaptive"
    if policy not in {"fixed", "schedule", "adaptive"}:
        raise ValueError(f"Unknown draft policy: {policy}")
    blocks = {name: getattr(args, name, None) for name in ("short_block", "medium_block", "long_block")}
    blocks = {name: args.block if value is None else value for name, value in blocks.items()}
    if any(value < 1 or value > args.block for value in blocks.values()):
        raise ValueError("Phase blocks must be in [1, --block]; block includes the known target token.")
    if policy == "fixed" and any(value != args.block for value in blocks.values()):
        raise ValueError("Phase block limits require --draft-policy schedule or adaptive.")
    medium_position = getattr(args, "medium_position", 96)
    long_position = getattr(args, "long_position", 192)
    if not 0 <= medium_position < long_position:
        raise ValueError("Require 0 <= medium-position < long-position.")
    runtime_mode = getattr(args, "runtime_mode", "diagnostic")
    if runtime_mode not in {"diagnostic", "throughput"}:
        raise ValueError(f"Unknown runtime mode: {runtime_mode}")
    settings = {
        "mode": "heuristic" if adaptive else policy,
        "max_block_tokens": args.block, **blocks,
        "medium_position": medium_position, "long_position": long_position,
        "compact_runtime_stats": runtime_mode == "throughput",
        "production_async_timing": runtime_mode == "throughput",
        "cheap_verifier_policy": "logit_margin" if adaptive else "none",
        "cheap_verifier_margin_skip_below": getattr(args, "target_skip_margin", 1.0) if adaptive else 0.0,
        "cheap_verifier_margin_block2_below": getattr(args, "target_short_margin", 3.0) if adaptive else 0.0,
        "min_draft_margin": getattr(args, "draft_min_margin", 1.0) if adaptive else 0.0,
        "draft_cooldown_after_failures": getattr(args, "cooldown_failures", 3) if adaptive else 0,
        "draft_cooldown_cycles": getattr(args, "cooldown_cycles", 4) if adaptive else 0,
    }
    if adaptive:
        if args.route == "ngram":
            raise ValueError("Adaptive T drafting requires tail or boundary route.")
        margins = [settings[name] for name in ("cheap_verifier_margin_skip_below",
                   "cheap_verifier_margin_block2_below", "min_draft_margin")]
        if any(not math.isfinite(value) or value < 0 for value in margins):
            raise ValueError("Adaptive margins must be finite and non-negative.")
        if 0 < margins[1] < margins[0]:
            raise ValueError("target-short-margin must be >= target-skip-margin, or 0 to disable the short-block gate.")
        failures, cycles = settings["draft_cooldown_after_failures"], settings["draft_cooldown_cycles"]
        if min(failures, cycles) < 0 or bool(failures) != bool(cycles):
            raise ValueError("Cooldown failures/cycles must both be positive, or both 0 to disable cooldown.")
    return settings


def command(args):
    settings = draft_policy_settings(args)
    source = "tail" if args.route == "ngram" else args.route
    cmd = [sys.executable, str(EXPERIMENT / "recurft_speculative_generate.py"),
           "--model-name-or-path", str(args.model), "--checkpoint", str(args.checkpoint),
           "--data-file", str(args.data), "--output-json", str(args.output / "result.json"),
           "--draft-commit-policy", "target_match", "--target-match-lambda", "1.0",
           "--ngram-draft-mode", "only" if args.route == "ngram" else "off",
           "--draft-logit-source", source, "--start-index", str(args.start), "--max-samples", str(args.samples),
           "--max-new-tokens", str(args.max_new_tokens), "--max-prompt-tokens", str(args.max_prompt_tokens),
           "--min-block-tokens", "1", "--dtype", args.dtype,
           "--device", args.device, "--question-prefix", "Solve the following math problem step by step.",
           "--question-suffix", "Put the final answer on its own line after 'Answer:'.",
           "--enable-thinking" if args.thinking else "--disable-thinking"]
    correction_mode = getattr(args, "correction_mode", "off")
    if correction_mode == "reuse":
        cmd.append("--reuse-verify-cache-for-correction")
    elif correction_mode == "defer":
        cmd.append("--defer-correction-to-next-verify")
    elif correction_mode != "off":
        raise ValueError(f"Unknown correction mode: {correction_mode}")
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--route", choices=["boundary", "tail", "ngram"], default="boundary")
    p.add_argument("--gpu", help="Physical GPU index or UUID, required for CUDA.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--block", type=int, default=3)
    p.add_argument("--draft-policy", choices=["fixed", "schedule", "adaptive"], default="fixed")
    for name in ("short-block", "medium-block", "long-block"):
        p.add_argument("--" + name, type=int, help="Phase block cap including the known token; defaults to --block.")
    p.add_argument("--medium-position", type=int, default=96)
    p.add_argument("--long-position", type=int, default=192)
    p.add_argument("--runtime-mode", choices=["diagnostic", "throughput"], default="diagnostic")
    p.add_argument("--target-skip-margin", type=float, default=1.0)
    p.add_argument("--target-short-margin", type=float, default=3.0)
    p.add_argument("--draft-min-margin", type=float, default=1.0)
    p.add_argument("--cooldown-failures", type=int, default=3)
    p.add_argument("--cooldown-cycles", type=int, default=4)
    p.add_argument("--correction-mode", choices=["off", "reuse", "defer"], default="off",
                   help="Reuse verified KV, defer correction to the next verification, or keep the original path.")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate assets and print command; do not launch a model.")
    args = p.parse_args()
    if args.device == "cuda" and args.gpu is None:
        p.error("--gpu is required for CUDA.")
    if args.device == "cpu" and (args.gpu is not None or args.dtype != "fp32"):
        p.error("CPU execution requires --dtype fp32 and no --gpu.")
    if min(args.max_new_tokens, args.max_prompt_tokens) <= 0 or args.block < 2:
        p.error("Token limits must be positive and block >= 2.")
    try:
        policy_settings = draft_policy_settings(args)
    except ValueError as e:
        p.error(str(e))
    for name in ("model", "checkpoint", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not (args.model / "config.json").is_file():
        p.error("--model must be a trusted local model directory, not an unchecked download.")
    metadata = checkpoint_metadata(args.checkpoint, args.route)
    coverage = validate_data(args.data, args.start, args.samples)
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        print(json.dumps({"coverage": coverage, "metadata": metadata}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "inputs.json", {"arguments": vars(args) | {k: str(getattr(args, k)) for k in
               ("model", "checkpoint", "data", "output")}, "data": coverage, "checkpoint": metadata})
    try:
        gpu = run_logged(cmd, args.output, args.gpu)
        audit = audit_result(args.output / "result.json", args.route, args.start, args.samples,
                             correction_mode=args.correction_mode, draft_policy=policy_settings)
        audit["timing_status"] = gpu["timing_status"]
        write_json(args.output / "complete.marker.json", audit)
        print(json.dumps(audit, indent=2))
    except Exception as e:
        write_json(args.output / "failed.marker.json", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
