#!/usr/bin/env python3
"""Compare serial-token and block-forward logits on the same token sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from recurft_rollout_eval import torch_dtype
from recurft_speculative_generate import load_rows, make_prompt, target_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--block-sizes", default="2,4,8")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--question-prefix", default="Solve the following math problem step by step.")
    parser.add_argument("--question-suffix", default="Put the final answer on its own line after 'Answer:'.")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def serial_trace(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[int], list[torch.Tensor], float]:
    start = time.perf_counter()
    outputs = target_forward(model, prompt_ids, start_position=0, output_hidden_states=False)
    cache = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]
    tokens: list[int] = []
    serial_next_logits: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        token = int(next_logits.argmax(dim=-1).item())
        tokens.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=prompt_ids.device)
        outputs = target_forward(
            model,
            token_tensor,
            past_key_values=cache,
            start_position=prompt_ids.size(1) + len(tokens) - 1,
            output_hidden_states=False,
        )
        cache = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        serial_next_logits.append(next_logits[0].float().detach())
    sync(prompt_ids.device)
    return tokens, serial_next_logits, time.perf_counter() - start


def margin_bin(value: float) -> str:
    if value < 0.05:
        return "<0.05"
    if value < 0.10:
        return "0.05-0.10"
    if value < 0.25:
        return "0.10-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 1.00:
        return "0.50-1.00"
    if value < 2.00:
        return "1.00-2.00"
    return ">=2.00"


@torch.no_grad()
def audit_block_size(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    tokens: list[int],
    serial_logits: list[torch.Tensor],
    block_size: int,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    outputs = target_forward(model, prompt_ids, start_position=0, output_hidden_states=False)
    cache = outputs.past_key_values
    rows: list[dict[str, Any]] = []
    for block_start in range(0, len(tokens), block_size):
        block_tokens = tokens[block_start : block_start + block_size]
        block_tensor = torch.tensor([block_tokens], dtype=torch.long, device=prompt_ids.device)
        outputs = target_forward(
            model,
            block_tensor,
            past_key_values=cache,
            start_position=prompt_ids.size(1) + block_start,
            output_hidden_states=False,
        )
        cache = outputs.past_key_values
        for offset in range(len(block_tokens)):
            token_index = block_start + offset
            if token_index >= len(serial_logits) or token_index + 1 >= len(tokens):
                continue
            serial = serial_logits[token_index]
            block = outputs.logits[0, offset].float()
            serial_logp = F.log_softmax(serial, dim=-1)
            block_logp = F.log_softmax(block, dim=-1)
            serial_prob = serial_logp.exp()
            top2 = torch.topk(serial, k=2).values
            margin = float((top2[0] - top2[1]).item())
            serial_top1 = int(serial.argmax().item())
            block_top1 = int(block.argmax().item())
            rows.append(
                {
                    "token_index": token_index + 1,
                    "teacher_token": tokens[token_index + 1],
                    "serial_top1": serial_top1,
                    "block_top1": block_top1,
                    "top1_match": serial_top1 == block_top1,
                    "serial_margin": margin,
                    "margin_bin": margin_bin(margin),
                    "serial_to_block_kl": float((serial_prob * (serial_logp - block_logp)).sum().item()),
                    "logit_max_abs": float((serial - block).abs().max().item()),
                    "logit_mean_abs": float((serial - block).abs().mean().item()),
                }
            )
    sync(prompt_ids.device)
    return rows, time.perf_counter() - start


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"positions": 0}
    mismatch = sum(not row["top1_match"] for row in rows)
    by_margin: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_margin.setdefault(row["margin_bin"], {"positions": 0, "mismatches": 0})
        bucket["positions"] += 1
        bucket["mismatches"] += int(not row["top1_match"])
    for bucket in by_margin.values():
        bucket["mismatch_rate"] = bucket["mismatches"] / max(1, bucket["positions"])
    return {
        "positions": len(rows),
        "top1_mismatches": mismatch,
        "top1_mismatch_rate": mismatch / len(rows),
        "mean_kl": sum(row["serial_to_block_kl"] for row in rows) / len(rows),
        "max_kl": max(row["serial_to_block_kl"] for row in rows),
        "mean_logit_abs": sum(row["logit_mean_abs"] for row in rows) / len(rows),
        "max_logit_abs": max(row["logit_max_abs"] for row in rows),
        "by_serial_margin": by_margin,
    }


def main() -> None:
    args = parse_args()
    block_sizes = [int(value) for value in args.block_sizes.split(",") if value.strip()]
    device = torch.device(args.device)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).to(device)
    model = PeftModel.from_pretrained(base_model, args.checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample_outputs = []
    aggregate: dict[int, list[dict[str, Any]]] = {block: [] for block in block_sizes}
    for sample_index, row in enumerate(load_rows(args.data_file, 0, args.max_samples)):
        prompt_ids = make_prompt(
            tokenizer,
            row["question"],
            args.question_prefix,
            args.question_suffix,
            True,
        )[:, -args.max_prompt_tokens :].to(device)
        tokens, serial_logits, serial_time = serial_trace(
            model,
            prompt_ids,
            args.max_new_tokens,
            tokenizer.eos_token_id,
        )
        sample = {"sample": sample_index, "generated_tokens": len(tokens), "serial_time_s": serial_time, "blocks": {}}
        for block_size in block_sizes:
            rows, block_time = audit_block_size(model, prompt_ids, tokens, serial_logits, block_size)
            for result in rows:
                result["sample"] = sample_index
                result["block_size"] = block_size
            aggregate[block_size].extend(rows)
            sample["blocks"][str(block_size)] = {**summarize(rows), "block_time_s": block_time}
        sample_outputs.append(sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)

    output = {
        "args": vars(args),
        "summary": {str(block): summarize(rows) for block, rows in aggregate.items()},
        "samples": sample_outputs,
        "rows": [row for rows in aggregate.values() for row in rows],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
