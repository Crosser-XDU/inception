#!/usr/bin/env python3
"""Materialize a portable staged training config and launch the bundled trainer."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import yaml

from common import ROOT, REPO, checkpoint_metadata, run_logged, write_json
from joint_training import joint_config

STAGES = {
    "core": "llama3_q3_recurft_1epoch.yaml",
    "multistep": "llama3_tail29_30_multistep_k4_huber_ms5e4_continue_20260708.yaml",
    "boundary-warmup": "llama3_q3_stage1_boundary_only_225_20260711.yaml",
    "boundary": "llama3_q3_stage1_boundary_only_continue_to_1000_20260711.yaml",
}
STARTS = {"multistep": 49375, "boundary-warmup": 59375, "boundary": 59600}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=[*STAGES, "joint"], required=True)
    for field in ("model", "data", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--source-config", type=Path, help="For joint: train.yaml that produced the boundary checkpoint.")
    p.add_argument("--template", help="Chat template override; use qwen3_nothink for non-thinking Qwen3.")
    p.add_argument("--gpu", required=True)
    p.add_argument("--steps", type=int, help="Override added steps; this is a new training recipe, not the frozen schedule.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for field in ("model", "data", "output", "checkpoint", "source_config"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    if not args.data.is_file() or not (args.model / "config.json").is_file():
        p.error("Supply an existing local base model and MetaMath training data file.")
    model_config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if args.stage == "joint":
        if args.checkpoint is None or args.source_config is None or args.steps is None:
            p.error("Joint requires --checkpoint, --source-config and --steps.")
        try:
            checkpoint_metadata(args.checkpoint, "boundary")
            source = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
            recipe = yaml.safe_load((ROOT / "configs/experimental/recurft_joint_boundary.yaml").read_text(encoding="utf-8"))
            config = joint_config(source, recipe, args.checkpoint, args.steps, model_config)
        except (OSError, ValueError, KeyError, TypeError) as error:
            p.error(str(error))
    else:
        if args.source_config is not None:
            p.error("--source-config is only used by --stage joint.")
        config = yaml.safe_load((ROOT / "configs/reference" / STAGES[args.stage]).read_text(encoding="utf-8"))
    if args.template is not None:
        config["template"] = args.template
    if model_config.get("model_type") == "qwen3" and config.get("template") not in ("qwen3", "qwen3_nothink"):
        p.error("Qwen3 requires --template qwen3_nothink (or qwen3 for reasoning data); llama3 is incompatible.")
    config.update(model_name_or_path=str(args.model), output_dir=str(args.output / "checkpoint_output"),
                  dataset_dir=str(args.output / "dataset"), dataset="metamathqa_full_sft",
                  overwrite_output_dir=False, report_to="none", plot_loss=False)
    config.pop("run_name", None)
    if args.stage != "joint":
        config.pop("resume_from_checkpoint", None)
    start = 0
    if args.stage not in ("core", "joint"):
        if args.checkpoint is None:
            p.error("Continuation requires --checkpoint, including trainer_state.json.")
        checkpoint_metadata(args.checkpoint, "boundary" if args.stage == "boundary" else "tail")
        start = json.loads((args.checkpoint / "trainer_state.json").read_text())["global_step"]
        if args.steps is None and start != STARTS[args.stage]:
            p.error(f"Frozen {args.stage} stage expects global step {STARTS[args.stage]}, found {start}.")
        config["resume_from_checkpoint"] = str(args.checkpoint)
    elif args.stage == "core" and args.checkpoint is not None:
        p.error("Core starts from the base model; checkpoint resumes are for continuation stages.")
    if args.steps is not None and args.stage != "joint":
        if args.steps <= 0:
            p.error("--steps must be positive.")
        config["max_steps"] = start + args.steps
    if args.output.exists():
        p.error("--output must be a new directory.")
    config_path = args.output / "train.yaml"
    cmd = [sys.executable, str(REPO / "src/train.py"), str(config_path)]
    print(yaml.safe_dump(config, sort_keys=False))
    print(shlex.join(cmd))
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "dataset/dataset_info.json", {"metamathqa_full_sft": {
        "file_name": str(args.data), "columns": {"prompt": "query", "response": "response"}}})
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        audit = run_logged(cmd, args.output, args.gpu)
        write_json(args.output / "complete.marker.json", {"stage": args.stage, "gpu_audit": audit,
                   "note": "Training process completed; evaluate the produced checkpoint separately."})
    except Exception as e:
        write_json(args.output / "failed.marker.json", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
