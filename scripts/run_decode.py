#!/usr/bin/env python3
"""Run one explicit, strict proposal route and verify what actually executed."""

import argparse
import json
from pathlib import Path
import shlex
import sys

from common import EXPERIMENT, audit_result, checkpoint_metadata, run_logged, validate_data, write_json


def command(args):
    source = "tail" if args.route == "ngram" else args.route
    cmd = [sys.executable, str(EXPERIMENT / "recurft_speculative_generate.py"),
           "--model-name-or-path", str(args.model), "--checkpoint", str(args.checkpoint),
           "--data-file", str(args.data), "--output-json", str(args.output / "result.json"),
           "--mode", "fixed", "--draft-commit-policy", "target_match", "--target-match-lambda", "1.0",
           "--ngram-draft-mode", "only" if args.route == "ngram" else "off",
           "--draft-logit-source", source, "--start-index", str(args.start), "--max-samples", str(args.samples),
           "--max-new-tokens", str(args.max_new_tokens), "--max-prompt-tokens", str(args.max_prompt_tokens),
           "--max-block-tokens", str(args.block), "--min-block-tokens", "1",
           "--short-block", str(args.block), "--medium-block", str(args.block), "--long-block", str(args.block),
           "--medium-position", "96", "--long-position", "192", "--dtype", args.dtype,
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
                             correction_mode=args.correction_mode)
        audit["timing_status"] = gpu["timing_status"]
        write_json(args.output / "complete.marker.json", audit)
        print(json.dumps(audit, indent=2))
    except Exception as e:
        write_json(args.output / "failed.marker.json", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
