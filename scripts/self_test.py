#!/usr/bin/env python3
"""Offline CPU tests: real tiny Llama, actual decoder, both neural readouts."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from common import ROOT, REPO, environment, sha256


def tiny_assets(directory):
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from llamafactory.model.model_utils.recurft import (
        RecurFTRecurrentModule, attach_recurft_recurrent_module, save_recurft_recurrent_module)

    torch.manual_seed(17)
    torch.set_num_threads(1)
    base, checkpoint = directory / "base", directory / "checkpoint"
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight[2].zero_()
    model.save_pretrained(base)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "The": 3, "sky": 4, "is": 5,
             "blue": 6, "today": 7, "Answer": 8, "2": 9, "4": 10}
    vocab.update({f"v{i}": i for i in range(11, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(checkpoint)
    metadata = dict(anchor_layer=0, last_layer=3, loop_start_layer=1, loop_end_layer=2,
                    loop_layer_ids=[1, 2], projection_layer_ids=[1, 2, 3],
                    recurrent_hidden_state_index=1, tail_layer_ids=None,
                    lora_target=["q_proj", "v_proj"], t_lora_rank=2, t_lora_alpha=4,
                    boundary_head_rank=4, token_conditioning_rank=0,
                    verifier_lora_rank=2, verifier_lora_alpha=4)
    recurrent = RecurFTRecurrentModule(layers=list(model.model.layers[1:3]),
                target_modules=metadata["lora_target"], rank=2, alpha=4, dropout=0,
                metadata=metadata, verifier_rank=2, verifier_alpha=4, verifier_dropout=0,
                verifier_probe_init_std=0.02, boundary_head_rank=4)
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                          task_type="CAUSAL_LM"))
    peft.save_pretrained(checkpoint)
    attach_recurft_recurrent_module(peft, recurrent)
    save_recurft_recurrent_module(peft, str(checkpoint))
    data = directory / "tiny.jsonl"
    data.write_text('\n'.join(json.dumps(x) for x in [
        {"question": "The sky is", "answer": "blue"},
        {"question": "2 2", "answer": "4"}]) + '\n')
    return base, checkpoint, data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, help="Persist the test report; tiny assets remain temporary.")
    args = p.parse_args()
    os.environ.update(environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(REPO / "experiments/recurft_math")]
    tests = ["model/model_utils/test_recurft.py", "train/test_recurft_multistep.py",
             "train/test_recurft_frozen_reference.py", "train/test_recurft_metric_logging.py",
             "test_recurft_verification_policy.py", "test_recurft_ngram_tree.py", "test_recurft_ngram_lookup.py"]
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="layerloop-unit-") as tmp:
        junit = Path(tmp) / "tests.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=",
                        "--junitxml", str(junit),
                        *[str(REPO / "tests" / name) for name in tests],
                        str(ROOT / "tests")], check=True)
        cases = ET.parse(junit).getroot().findall(".//testcase")
        if not cases or any(list(case) for case in cases):
            raise RuntimeError("Tests must execute without skips, failures, or errors.")
        passed = [f"{case.get('classname')}:{case.get('name')}" for case in cases]
    runs = []
    with tempfile.TemporaryDirectory(prefix="layerloop-cpu-") as tmp:
        tmp = Path(tmp)
        base, checkpoint, data = tiny_assets(tmp)
        for route in ("boundary", "tail", "ngram"):
            out = tmp / route
            cmd = [sys.executable, str(ROOT / "scripts/run_decode.py"), "--model", str(base),
                   "--checkpoint", str(checkpoint), "--data", str(data), "--output", str(out),
                   "--route", route, "--device", "cpu", "--dtype", "fp32", "--samples", "2",
                   "--max-new-tokens", "8", "--max-prompt-tokens", "64"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode:
                print(result.stdout, result.stderr)
                if (out / "run.log").exists():
                    print((out / "run.log").read_text())
                raise RuntimeError(f"Tiny {route} decode failed.")
            runs.append(json.loads((out / "complete.marker.json").read_text()))
    report = {"status": "passed", "unit_tests": passed, "tiny_decode_runs": runs,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "tested_files": {str(p.relative_to(ROOT)): sha256(p)
                               for base in (REPO, ROOT / "scripts", ROOT / "tests", ROOT / "configs")
                               for p in sorted(base.rglob("*"))
                               if p.is_file() and p.suffix in (".py", ".sh", ".yaml")},
              "environment": {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "pytest")},
              "scope": "Random tiny CPU model; validates execution and routing, not 8B quality/speed reproduction."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({"status": "passed", "unit_tests": len(passed), "tiny_routes": [r["route"] for r in runs]}))


if __name__ == "__main__":
    main()
