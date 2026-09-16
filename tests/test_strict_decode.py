"""Real torch acceptance/control-flow tests, using deterministic toy target/T modules.

Run on CPU without model weights; the CUDA overlap test also runs on a GPU host.
"""
import argparse
import ast
from bisect import bisect_right
import itertools
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

DECODER = Path(__file__).resolve().parents[1] / "LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py"


class ToyCache:
    def __init__(self):
        self.layers = []

    def update(self, keys, values, layer_idx):
        while len(self.layers) <= layer_idx:
            self.layers.append(SimpleNamespace(keys=None, values=None))
        layer = self.layers[layer_idx]
        layer.keys = keys if layer.keys is None else torch.cat((layer.keys, keys), dim=-2)
        layer.values = values if layer.values is None else torch.cat((layer.values, values), dim=-2)
        return layer.keys, layer.values

    def crop(self, length):
        for layer in self.layers:
            if layer.keys is not None:
                layer.keys = layer.keys[..., :length, :]
                layer.values = layer.values[..., :length, :]

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.layers) or self.layers[layer_idx].keys is None:
            return 0
        return self.layers[layer_idx].keys.size(-2)


def logits_for(ids):
    logits = torch.full((*ids.shape, 17), -10.0, device=ids.device)
    return logits.scatter(-1, ids.long().remainder(17).unsqueeze(-1), 10.0)


class ToyT:
    def __init__(self, error_mode=0):
        self.error_mode = error_mode

    def has_token_conditioning(self):
        return False

    def forward_with_cache(self, hidden, model, past_key_value, token_ids, rollout_step=1):
        past_key_value.update(hidden.unsqueeze(1), hidden.unsqueeze(1), 0)
        error = ((hidden.long() % 4 == 1).to(hidden.dtype) if self.error_mode == 1
                 else torch.full_like(hidden, 2) if self.error_mode == 2 else 0)
        return (hidden + 1 + error).remainder(17), past_key_value

    def boundary_logits(self, hidden, model):
        return logits_for(hidden[..., 0] + 1)


def target_forward(model, ids, *, past_key_values=None, start_position=0, **kwargs):
    cache = past_key_values if past_key_values is not None else ToyCache()
    cache.crop(start_position)
    assert cache.get_seq_length() == start_position, (cache.get_seq_length(), start_position)
    hidden = ids.float().unsqueeze(-1)
    cache.update(hidden.unsqueeze(1), hidden.unsqueeze(1), 0)
    return SimpleNamespace(logits=logits_for(ids + 1), hidden_states={1: hidden}, past_key_values=cache)


def decoder_namespace():
    tree = ast.parse(DECODER.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ns = dict(torch=torch, argparse=argparse, math=math, re=re, time=time, Path=Path,
              SimpleNamespace=SimpleNamespace, bisect_right=bisect_right, DynamicCache=ToyCache,
              _SYNC_COMPONENT_TIMING=False)
    exec(compile(ast.fix_missing_locations(module), str(DECODER), "exec"), ns)
    ns.update(target_forward=target_forward, get_base_causal_lm=lambda model: model)
    return ns


@unittest.skipIf(torch is None, "torch is required for tensor and decoder control-flow tests")
class StrictDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = decoder_namespace()

    def args(self, **changes):
        argv = ["decoder", "--model-name-or-path", "toy", "--checkpoint", "toy", "--data-file", "toy",
                "--output-json", "toy", "--mode", "fixed", "--draft-logit-source", "boundary",
                "--compact-runtime-stats", "--max-new-tokens", "13", "--max-block-tokens", "3",
                "--lazy-t-sync", "--defer-t-init-until-latent"]
        with patch.object(sys, "argv", argv):
            args = self.ns["parse_args"]()
        for key, value in changes.items():
            setattr(args, key, value)
        self.ns["validate_fast_strict_verification"](args)
        return args

    def test_all_prefix_patterns_ties_and_matched_tokens_after_rejection(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for length in range(6):
                for pattern in itertools.product((False, True), repeat=length):
                    targets = torch.arange(length).remainder(3)
                    drafts = torch.tensor([int(t) if match else (int(t) + 1) % 3
                                           for t, match in zip(targets, pattern)], dtype=torch.long)
                    logits = torch.zeros((1, length + 1, 3), dtype=dtype)
                    for i, token in enumerate(targets):
                        logits[0, i, token] = 4
                    block = torch.cat((torch.tensor([2]), drafts)).reshape(1, -1)
                    prefix = next((i for i, match in enumerate(pattern) if not match), length)
                    actual = self.ns["strict_verify_prefix"](logits, block)
                    self.assertEqual(actual, (prefix + 1, int(targets[prefix]) if prefix < length else None, sum(pattern)))
            tied = torch.zeros((1, 3, 3), dtype=dtype)
            self.assertEqual(self.ns["strict_verify_prefix"](tied, torch.tensor([[2, 0, 1]])), (2, 0, 1))

    def test_policy_guard_refuses_relaxed_or_diagnostic_dependent_decisions(self):
        cases = [dict(target_match_lambda=0.99), dict(target_match_lambda_late=0.5),
                 dict(latent_draft_commit_policy="whole_block"), dict(mode="heuristic"),
                 dict(compact_runtime_stats=False), dict(ngram_draft_mode="only"),
                 dict(draft_token_category_gate="number"), dict(target_match_category_lambdas="number:0.1"),
                 dict(serial_fallback_margin=0.1), dict(precommit_unchecked_drafts=True),
                 dict(preemptive_lookahead=True), dict(verification_frequency_policy="agreement_gated")]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.args(fast_strict_verification=True, **changes)

    def decode(self, args, error_mode, eos, device="cpu"):
        return self.ns["speculative_decode"](object(), ToyT(error_mode),
            {"anchor_layer": 0, "recurrent_hidden_state_index": 1, "projection_layer_ids": [0], "loop_start_layer": 0},
            torch.tensor([[0, 1]], device=device), args, eos, torch.device(device))

    def assert_same_decode(self, old, new):
        for key in ("token_ids", "target_calls", "draft_tokens", "accepted_draft_tokens", "matched_draft_tokens",
                    "target_match_observations", "accepted_mismatch_tokens", "accepted_unchecked_tokens",
                    "block_hist", "accepted_hist", "block_hist_by_phase", "accepted_hist_by_phase",
                    "deferred_corrective_tokens", "fast_correction_cache_reuses", "target_generated_positions"):
            self.assertEqual(old[key], new[key], key)
        fields = ("generated_start", "proposed_len", "accepted_len", "accelerated_generated_positions",
                  "accelerated_sequence_positions", "committed_corrective_token", "corrective_deferred")
        self.assertEqual([{k:b.get(k) for k in fields} for b in old["block_records"]],
                         [{k:b.get(k) for k in fields} for b in new["block_records"]])
        self.assertEqual(new["fast_strict_blocks"], len(new["block_records"]))
        self.assertEqual(new["fast_strict_draft_tokens"], new["draft_tokens"])

    def test_full_decoder_rejection_eos_budget_cache_and_correction_equivalence(self):
        for error, correction, block, batched in itertools.product(range(3), ("off", "reuse", "defer"), (2, 3, 5), (False, True)):
            for start, eos in ((0, None), (0, 2), (0, 3), (0, 6), (4, 6), (99, None)):
                with self.subTest(error=error, correction=correction, block=block, batched=batched, start=start, eos=eos):
                    args = self.args(max_block_tokens=block, batched_draft_boundary=batched,
                        latent_draft_min_position=start, reuse_verify_cache_for_correction=correction=="reuse",
                        defer_correction_to_next_verify=correction=="defer", inplace_draft_cache=correction!="off")
                    old = self.decode(args, error, eos)
                    args.fast_strict_verification = True
                    new = self.decode(args, error, eos)
                    self.assert_same_decode(old, new)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required for real stream overlap")
    def test_cuda_stream_overlap_matches_serial_drafting(self):
        for error, fast in itertools.product(range(3), (False, True)):
            args = self.args(fast_strict_verification=fast, defer_correction_to_next_verify=True,
                             inplace_draft_cache=True, max_block_tokens=5)
            serial = self.decode(args, error, 11, "cuda")
            args.single_gpu_parallel_draft = True
            parallel = self.decode(args, error, 11, "cuda")
            for key in ("token_ids", "target_calls", "block_hist", "accepted_hist", "accepted_draft_tokens"):
                self.assertEqual(serial[key], parallel[key])
            self.assertGreater(parallel["parallel_draft_steps"], 0)


if __name__ == "__main__":
    unittest.main()
