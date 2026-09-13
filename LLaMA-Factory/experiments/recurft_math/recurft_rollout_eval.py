#!/usr/bin/env python3
"""Evaluate RecurFT latent rollout drift and rough speed proxies.

This script is intentionally diagnostic. It does not replace normal generation.
It measures whether the learned T module can cheaply approximate future anchor
hidden states, and how the projected logits compare with full-model logits.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.model.model_utils.recurft import (
    RECURFT_CONFIG_NAME,
    RECURFT_SAFE_WEIGHTS_NAME,
    RECURFT_WEIGHTS_NAME,
    RecurFTRecurrentModule,
    find_decoder_layers,
)

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover
    safe_load_file = None


DEFAULT_MODEL = os.environ.get("RECURFT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
DEFAULT_OUTPUT = os.environ.get("RECURFT_CHECKPOINT", "outputs/recurft-checkpoint")
DEFAULT_DATA = os.environ.get("RECURFT_DATA", "data/MetaMathQA.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_OUTPUT, help="Checkpoint dir or an output dir containing checkpoint-*.")
    parser.add_argument("--data-file", default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--prefix-tokens", type=int, default=192)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--logit-source",
        choices=["auto", "boundary", "tail"],
        default="auto",
        help=(
            "Project rollout states with the cheap boundary head, the full post-T tail, "
            "or preserve the legacy auto behavior (boundary when available)."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["raw", "chat_template"],
        default="raw",
        help="Use raw prompt+response text or tokenizer chat template formatting.",
    )
    parser.add_argument(
        "--multistep-residual-start-step-override",
        type=int,
        default=None,
        help="Diagnostic-only override for the recurrent residual start step.",
    )
    return parser.parse_args()


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if (path / RECURFT_CONFIG_NAME).exists():
        return path

    checkpoints = []
    for child in path.glob("checkpoint-*"):
        try:
            step = int(child.name.rsplit("-", maxsplit=1)[-1])
        except ValueError:
            continue
        if (child / RECURFT_CONFIG_NAME).exists():
            checkpoints.append((step, child))

    if not checkpoints:
        raise FileNotFoundError(f"No RecurFT checkpoint found under {path}.")

    return max(checkpoints, key=lambda item: item[0])[1]


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def get_base_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def get_decoder(model: torch.nn.Module) -> torch.nn.Module:
    base = get_base_causal_lm(model)
    return getattr(base, "model", base)


def make_causal_mask(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    min_dtype = torch.finfo(hidden_states.dtype).min
    mask = torch.full((seq_len, seq_len), min_dtype, dtype=hidden_states.dtype, device=hidden_states.device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len).clone()
    if attention_mask is not None:
        mask = mask.masked_fill(attention_mask[:, None, None, :].eq(0), min_dtype)
    return mask


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def maybe_position_embeddings(model: torch.nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    decoder = get_decoder(model)
    rotary_emb = getattr(decoder, "rotary_emb", None)
    if rotary_emb is None:
        return None
    try:
        return rotary_emb(hidden_states, position_ids)
    except TypeError:
        return None


def call_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    model: torch.nn.Module,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "attention_mask": make_causal_mask(hidden_states, attention_mask),
        "position_ids": position_ids,
        "output_attentions": False,
        "use_cache": False,
        "cache_position": torch.arange(hidden_states.size(1), device=hidden_states.device, dtype=torch.long),
    }
    position_embeddings = maybe_position_embeddings(model, hidden_states, position_ids)
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    signature = inspect.signature(layer.forward)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    try:
        output = layer(hidden_states, **kwargs)
    except TypeError:
        kwargs.pop("position_embeddings", None)
        output = layer(hidden_states, **kwargs)

    return output[0] if isinstance(output, tuple) else output


def tail_logits_from_anchor(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    anchor_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    base = get_base_causal_lm(model)
    layers, _ = find_decoder_layers(base)
    position_ids = make_position_ids(attention_mask)
    hidden_states = anchor_hidden
    projection_layer_ids = metadata.get("projection_layer_ids") or metadata.get("tail_layer_ids") or [metadata["last_layer"]]
    for layer_idx in projection_layer_ids:
        hidden_states = call_layer(layers[layer_idx], hidden_states, base, attention_mask, position_ids)

    decoder = get_decoder(base)
    norm = getattr(decoder, "norm", None) or getattr(decoder, "ln_f", None)
    if norm is not None:
        hidden_states = norm(hidden_states)

    lm_head = base.get_output_embeddings()
    return lm_head(hidden_states)


def load_records(path: str | Path, max_samples: int) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return [
            {
                "prompt": "Solve the problem step by step: What is 17 times 23?",
                "response": "17 times 23 is 17 times 20 plus 17 times 3, which is 340 plus 51, so the answer is 391.",
            }
        ][:max_samples]

    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    else:
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)

    records = []
    for item in rows:
        prompt = item.get("query") or item.get("instruction") or item.get("prompt") or item.get("question") or ""
        response = (
            item.get("response")
            or item.get("output")
            or item.get("baseline_response")
            or item.get("answer")
            or ""
        )
        if prompt and response:
            records.append({"prompt": str(prompt), "response": str(response)})
    return records


def encode_record(
    tokenizer,
    record: dict[str, str],
    max_length: int,
    device: torch.device,
    input_format: str = "raw",
) -> torch.Tensor:
    if input_format == "chat_template":
        messages = [
            {"role": "user", "content": record["prompt"].strip()},
            {"role": "assistant", "content": record["response"].strip()},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        add_special_tokens = False
    else:
        text = record["prompt"].strip() + "\n" + record["response"].strip()
        add_special_tokens = True

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    return encoded["input_ids"].to(device)


def build_recurrent_module(base_model: torch.nn.Module, checkpoint: Path, metadata: dict[str, Any]) -> RecurFTRecurrentModule:
    layers, _ = find_decoder_layers(base_model)
    loop_layer_ids = metadata.get("loop_layer_ids") or metadata.get("tail_layer_ids") or [metadata["last_layer"]]
    module = RecurFTRecurrentModule(
        layers=[layers[idx] for idx in loop_layer_ids],
        target_modules=metadata["lora_target"],
        rank=metadata["t_lora_rank"],
        alpha=metadata["t_lora_alpha"],
        dropout=0.0,
        metadata=metadata,
        verifier_rank=metadata.get("verifier_lora_rank", metadata["t_lora_rank"]),
        verifier_alpha=metadata.get("verifier_lora_alpha", metadata["t_lora_alpha"]),
        verifier_dropout=0.0,
        verifier_probe_init_std=metadata.get("verifier_probe_init_std", 0.02),
        boundary_head_rank=metadata.get("boundary_head_rank", 0),
        token_conditioning_rank=metadata.get("token_conditioning_rank", 0),
        multistep_residual_rank=metadata.get("multistep_residual_rank", 0),
        multistep_residual_alpha=metadata.get("multistep_residual_alpha", 0),
        multistep_residual_start_step=metadata.get("multistep_residual_start_step", 2),
        multistep_step1_residual_rank=metadata.get("multistep_step1_residual_rank", 0),
        multistep_step1_residual_alpha=metadata.get("multistep_step1_residual_alpha", 0),
    )

    safe_path = checkpoint / RECURFT_SAFE_WEIGHTS_NAME
    bin_path = checkpoint / RECURFT_WEIGHTS_NAME
    if safe_path.exists() and safe_load_file is not None:
        state_dict = safe_load_file(str(safe_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"Missing recurrent weights in {checkpoint}.")
    module.load_state_dict(state_dict, strict=False)
    return module


@torch.no_grad()
def recurrent_rollout(
    recurrent: RecurFTRecurrentModule,
    model_for_rotary: torch.nn.Module,
    prefix_hidden: torch.Tensor,
    steps: int,
    prefix_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_mask = torch.ones(prefix_hidden.shape[:2], dtype=torch.long, device=prefix_hidden.device)
    hidden_seq = prefix_hidden
    token_ids = prefix_token_ids
    preds = []
    for step in range(1, steps + 1):
        pred_seq = recurrent(
            hidden_seq,
            attention_mask=attention_mask,
            model=model_for_rotary,
            token_ids=token_ids,
            rollout_step=step,
        )
        next_hidden = pred_seq[:, -1:, :]
        preds.append(next_hidden)
        if recurrent.has_token_conditioning():
            if not recurrent.has_boundary_head():
                raise ValueError("Token-conditioned rollout requires the RecurFT boundary head.")
            next_token = recurrent.boundary_logits(hidden_seq[:, -1:, :], model_for_rotary).argmax(dim=-1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        hidden_seq = torch.cat([hidden_seq, next_hidden], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)
    return torch.cat(preds, dim=1), hidden_seq


def time_call(fn, iters: int, device: torch.device) -> float:
    for _ in range(max(1, min(2, iters))):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / max(1, iters)


@torch.no_grad()
def time_full_decode_steps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefix_len: int,
    steps: int,
    device: torch.device,
    iters: int,
) -> float | None:
    if input_ids.size(1) < prefix_len + steps:
        return None

    def run_once():
        prefix = input_ids[:, :prefix_len]
        out = model(prefix, use_cache=True, return_dict=True)
        past = out.past_key_values
        for step in range(steps):
            token = input_ids[:, prefix_len + step : prefix_len + step + 1]
            out = model(token, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values

    try:
        return time_call(run_once, iters, device)
    except Exception as exc:
        print(json.dumps({"full_decode_timing_error": str(exc)}, ensure_ascii=False))
        return None


def summarize(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def consecutive_true(values: torch.Tensor) -> int:
    length = 0
    for value in values.flatten().tolist():
        if not value:
            break
        length += 1
    return length


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    data_path = Path(args.data_file).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Evaluation data file does not exist: {data_path}")
    data_sha256 = sha256_file(data_path)
    data_bytes = data_path.stat().st_size
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    with (checkpoint / RECURFT_CONFIG_NAME).open(encoding="utf-8") as f:
        metadata = json.load(f)
    if args.multistep_residual_start_step_override is not None:
        if args.multistep_residual_start_step_override < 1:
            raise ValueError("--multistep-residual-start-step-override must be at least 1.")
        metadata["multistep_residual_start_step"] = args.multistep_residual_start_step_override

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(device)
    recurrent = build_recurrent_module(base_model, checkpoint, metadata).to(device).eval()
    model = PeftModel.from_pretrained(base_model, checkpoint).to(device).eval()

    records = load_records(data_path, args.max_samples)
    results: list[dict[str, float]] = []
    step_results: list[dict[str, list[float] | int]] = []
    speed_results: list[dict[str, float]] = []
    if args.logit_source == "boundary" and not recurrent.has_boundary_head():
        raise ValueError("--logit-source boundary requires a trained boundary head.")
    resolved_logit_source = (
        "boundary"
        if args.logit_source == "boundary"
        or (args.logit_source == "auto" and recurrent.has_boundary_head())
        else "tail"
    )

    for idx, record in enumerate(records):
        if len(results) >= args.max_samples:
            break
        input_ids = encode_record(tokenizer, record, args.max_length, device, args.input_format)
        needed = args.prefix_tokens + args.rollout_steps + 1
        if input_ids.size(1) < needed:
            continue

        input_ids = input_ids[:, : max(args.max_length, needed)]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        anchor_idx = metadata.get("recurrent_hidden_state_index", metadata["anchor_layer"] + 1)
        anchor_hidden = outputs.hidden_states[anchor_idx]
        prefix_hidden = anchor_hidden[:, : args.prefix_tokens, :]
        target_hidden = anchor_hidden[:, args.prefix_tokens : args.prefix_tokens + args.rollout_steps, :]
        pred_hidden, pred_sequence = recurrent_rollout(
            recurrent,
            get_base_causal_lm(model),
            prefix_hidden,
            args.rollout_steps,
            prefix_token_ids=input_ids[:, : args.prefix_tokens],
        )

        per_step_hidden_mse = (pred_hidden.float() - target_hidden.float()).square().mean(dim=-1).squeeze(0)
        per_step_cosine = F.cosine_similarity(pred_hidden.float(), target_hidden.float(), dim=-1).squeeze(0)
        hidden_mse = per_step_hidden_mse.mean().item()
        cosine = per_step_cosine.mean().item()

        pred_attention = torch.ones(pred_sequence.shape[:2], dtype=torch.long, device=device)
        if resolved_logit_source == "boundary":
            pred_logits = recurrent.boundary_logits(pred_sequence, get_base_causal_lm(model))
        else:
            pred_logits = tail_logits_from_anchor(model, metadata, pred_sequence, pred_attention)
        pred_logits = pred_logits[:, args.prefix_tokens : args.prefix_tokens + args.rollout_steps, :]
        teacher_logits = outputs.logits[:, args.prefix_tokens : args.prefix_tokens + args.rollout_steps, :]
        per_step_kl = F.kl_div(
            F.log_softmax(pred_logits.float(), dim=-1),
            F.softmax(teacher_logits.float(), dim=-1),
            reduction="none",
        ).sum(dim=-1).squeeze(0)
        kl = per_step_kl.mean().item()
        teacher_match = pred_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
        top1_agree = teacher_match.float().mean().item()

        gold_next = input_ids[:, args.prefix_tokens + 1 : args.prefix_tokens + args.rollout_steps + 1]
        gold_match = pred_logits.argmax(dim=-1) == gold_next
        gold_top1 = gold_match.float().mean().item()
        results.append(
            {
                "sample": float(idx),
                "hidden_mse": hidden_mse,
                "hidden_cosine": cosine,
                "logit_kl": kl,
                "teacher_top1_agreement": top1_agree,
                "gold_next_top1": gold_top1,
            }
        )
        step_results.append(
            {
                "hidden_mse": per_step_hidden_mse.tolist(),
                "hidden_cosine": per_step_cosine.tolist(),
                "logit_kl": per_step_kl.tolist(),
                "teacher_match": teacher_match.float().squeeze(0).tolist(),
                "gold_match": gold_match.float().squeeze(0).tolist(),
                "teacher_safe_len": consecutive_true(teacher_match),
                "gold_safe_len": consecutive_true(gold_match),
            }
        )

        if not args.skip_speed:
            full_decode_time = time_full_decode_steps(
                model,
                input_ids,
                args.prefix_tokens,
                args.rollout_steps,
                device,
                args.benchmark_iters,
            )

            def time_latent_only():
                recurrent_rollout(
                    recurrent,
                    get_base_causal_lm(model),
                    prefix_hidden,
                    args.rollout_steps,
                    prefix_token_ids=input_ids[:, : args.prefix_tokens],
                )

            latent_time = time_call(time_latent_only, args.benchmark_iters, device)

            def time_latent_plus_tail():
                _, seq = recurrent_rollout(
                    recurrent,
                    get_base_causal_lm(model),
                    prefix_hidden,
                    args.rollout_steps,
                    prefix_token_ids=input_ids[:, : args.prefix_tokens],
                )
                mask = torch.ones(seq.shape[:2], dtype=torch.long, device=device)
                tail_logits_from_anchor(model, metadata, seq, mask)

            latent_tail_time = time_call(time_latent_plus_tail, args.benchmark_iters, device)
            item = {
                "sample": float(idx),
                "latent_rollout_s": latent_time,
                "latent_plus_tail_s": latent_tail_time,
            }
            if recurrent.has_boundary_head():
                def time_latent_plus_boundary():
                    _, seq = recurrent_rollout(
                        recurrent,
                        get_base_causal_lm(model),
                        prefix_hidden,
                        args.rollout_steps,
                        prefix_token_ids=input_ids[:, : args.prefix_tokens],
                    )
                    recurrent.boundary_logits(seq, get_base_causal_lm(model))

                latent_boundary_time = time_call(time_latent_plus_boundary, args.benchmark_iters, device)
                item["latent_plus_boundary_s"] = latent_boundary_time
            if full_decode_time is not None:
                item["full_decode_s"] = full_decode_time
                item["speedup_latent_only"] = full_decode_time / latent_time
                item["speedup_latent_plus_tail"] = full_decode_time / latent_tail_time
                if recurrent.has_boundary_head():
                    item["speedup_latent_plus_boundary"] = full_decode_time / latent_boundary_time
            speed_results.append(item)

        print(json.dumps({"sample_result": results[-1]}, ensure_ascii=False))
        if speed_results:
            print(json.dumps({"sample_speed": speed_results[-1]}, ensure_ascii=False))

    summary = {
        "checkpoint": str(checkpoint),
        "model_name_or_path": str(args.model_name_or_path),
        "data_file": str(data_path),
        "data_sha256": data_sha256,
        "data_bytes": data_bytes,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "samples": len(results),
        "prefix_tokens": args.prefix_tokens,
        "rollout_steps": args.rollout_steps,
        "multistep_residual_start_step": metadata.get("multistep_residual_start_step", 2),
        "input_format": args.input_format,
        "logit_source": resolved_logit_source,
        "hidden_mse": summarize([x["hidden_mse"] for x in results]),
        "hidden_cosine": summarize([x["hidden_cosine"] for x in results]),
        "logit_kl": summarize([x["logit_kl"] for x in results]),
        "teacher_top1_agreement": summarize([x["teacher_top1_agreement"] for x in results]),
        "gold_next_top1": summarize([x["gold_next_top1"] for x in results]),
    }
    if step_results:
        summary["per_step"] = [
            {
                "step": step + 1,
                "hidden_mse": summarize([x["hidden_mse"][step] for x in step_results]),
                "hidden_cosine": summarize([x["hidden_cosine"][step] for x in step_results]),
                "logit_kl": summarize([x["logit_kl"][step] for x in step_results]),
                "teacher_top1": summarize([x["teacher_match"][step] for x in step_results]),
                "gold_top1": summarize([x["gold_match"][step] for x in step_results]),
            }
            for step in range(args.rollout_steps)
        ]
        for key in ["teacher_safe_len", "gold_safe_len"]:
            values = [int(x[key]) for x in step_results]
            summary[key] = {
                "mean": summarize(values),
                "max": max(values),
                "hist": {str(step): values.count(step) for step in range(args.rollout_steps + 1)},
                "survival": {
                    str(step): summarize([float(value >= step) for value in values])
                    for step in range(1, args.rollout_steps + 1)
                },
            }
    if speed_results:
        for key in ["full_decode_s", "latent_rollout_s", "latent_plus_tail_s", "speedup_latent_only", "speedup_latent_plus_tail"]:
            values = [x[key] for x in speed_results if key in x]
            if values:
                summary[key] = summarize(values)

    sample_records = []
    for result, step_result in zip(results, step_results):
        sample_records.append(
            {
                **result,
                "sample": int(result["sample"]),
                "teacher_safe_len": int(step_result["teacher_safe_len"]),
                "gold_safe_len": int(step_result["gold_safe_len"]),
                "teacher_match": [int(value) for value in step_result["teacher_match"]],
                "gold_match": [int(value) for value in step_result["gold_match"]],
            }
        )
    output = {"summary": summary, "samples": sample_records}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
