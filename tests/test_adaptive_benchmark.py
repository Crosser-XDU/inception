"""Exercise adaptive launch arguments and result accounting without model weights."""

import argparse
import ast
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_cache_benchmark as cache_tests
from common import audit_draft_policy, audit_result
from run_decode import command, draft_policy_settings

ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py"


def decoder_functions():
    # Execute the real parser and pure gate, without importing torch/transformers.
    tree = ast.parse(DECODER.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"parse_args", "cheap_verifier_decision", "parse_category_set", "choose_block_limit"}]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *selected], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(ast.fix_missing_locations(module), str(DECODER), "exec"), namespace)
    return namespace


def parse_decoder(cmd):
    with patch.object(sys, "argv", cmd[1:]):
        return decoder_functions()["parse_args"]()


class AdaptiveBenchmarkTests(unittest.TestCase):
    def args(self):
        args = cache_tests.CorrectionModeTests().args("reuse")
        args.draft_policy = "adaptive"
        args.block = 5
        return args

    def record(self, zero=False):
        expected = draft_policy_settings(self.args())
        hist = {"1": 2} if zero else {"1": 2, "3": 2}
        counters = {"cheap_policy_evaluations": sum(hist.values()), "cheap_policy_skips": 2,
                    "draft_cooldown_skips": 1, "draft_cooldown_activations": 1}
        summary = {"args": expected | {"draft_commit_policy": "target_match", "target_match_lambda": 1.0,
                   "ngram_draft_mode": "off", "draft_logit_source": "tail"},
                   "samples": 1, "draft_tokens": 0 if zero else 4, "accepted_draft_tokens": 0 if zero else 3,
                   "adaptive_timing_sums": {"t_step_s": 0 if zero else 1, "tail_s": 0 if zero else 1,
                                            "boundary_s": 0},
                   "accepted_mismatch_tokens": 0, "accepted_unchecked_tokens": 0,
                   "wall_clock_speedup": 1.0, "target_call_speedup": 1.0, **counters}
        rows = [{"sample": 0, "baseline_correct": True, "correct": True,
                 "adaptive": {"block_hist": hist, "accepted_mismatch_tokens": 0,
                              "accepted_unchecked_tokens": 0, **counters}}]
        return summary, rows, expected

    def test_real_decoder_accepts_policy_and_raw_logit_threshold_boundaries(self):
        args = self.args()
        parsed = parse_decoder(command(args))
        self.assertEqual(parsed.draft_commit_policy, "target_match")
        self.assertEqual(parsed.target_match_lambda, 1.0)
        self.assertEqual(parsed.ngram_draft_mode, "off")
        self.assertTrue(parsed.reuse_verify_cache_for_correction)
        for key, value in draft_policy_settings(args).items():
            self.assertEqual(getattr(parsed, key), value)
        gate = decoder_functions()["cheap_verifier_decision"]
        for margin, block, skip in [(0.99, 1, True), (1.0, 2, False), (2.99, 2, False), (3.0, 5, False)]:
            with self.subTest(margin=margin):
                result = gate(parsed, 0, None, 5, target_top1_margin=margin)
                self.assertEqual((result["block_limit"], result["skip"]), (block, skip))

    def test_summary_counts_drafts_without_anchor_or_double_counting_skips(self):
        summary, rows, expected = self.record()
        result = audit_draft_policy(summary, rows, expected)
        self.assertEqual(result["target_only_blocks"], 2)  # not 2 gate + 1 cooldown
        self.assertEqual(result["target_only_block_rate"], 0.5)
        self.assertEqual(result["mean_accepted_drafts_per_draft_block"], 1.5)
        self.assertEqual(result["mean_accepted_drafts_per_block"], 0.75)

    def test_async_timing_does_not_report_gpu_decode_only_from_host_components(self):
        tree = ast.parse(DECODER.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        guard = next(node for node in main.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Attribute) and node.test.attr == "production_async_timing"
                     and any(isinstance(child, ast.Assign) and any(isinstance(t, ast.Name)
                             and t.id == "parallelizable_aux_time" for t in child.targets) for child in node.body))
        namespace = {"args": argparse.Namespace(production_async_timing=True),
                     "baseline_decode_only_time": 8.0, "adaptive_decode_only_time": 6.0,
                     "adaptive_steady_decode_time": 5.0}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(DECODER), "exec"), namespace)
        for name in ("baseline_decode_only_time", "adaptive_decode_only_time", "adaptive_steady_decode_time"):
            self.assertIsNone(namespace[name])

    def test_phase_stats_exclude_anchor_and_report_reaching_samples(self):
        summary, rows, expected = self.record()
        rows[0]["adaptive"].update(
            block_hist_by_phase={"early": {"1": 2}, "late": {"3": 2}},
            accepted_hist_by_phase={"early": {"1": 2}, "late": {"2": 1, "3": 1}})
        phases = audit_draft_policy(summary, rows, expected)["phase_draft_stats"]
        self.assertEqual(phases["early"]["accepted_draft_tokens"], 0)
        self.assertEqual(phases["late"]["mean_accepted_drafts_per_draft_block"], 1.5)
        self.assertEqual(phases["late"]["reaching_samples"], 1)
        self.assertNotIn("mid", phases)

    def test_all_skipped_is_reported_but_not_as_recurrent_execution(self):
        summary, rows, expected = self.record(zero=True)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            p.write_text(json.dumps({"summary": summary, "results": rows}))
            result = audit_result(p, "tail", 0, 1, draft_policy=expected)
            self.assertFalse(result["recurrent_drafting_observed"])
            self.assertIsNone(result["mean_accepted_drafts_per_draft_block"])
            self.assertEqual(result["target_only_block_rate"], 1.0)
            with self.assertRaisesRegex(ValueError, "No evidence"):
                audit_result(p, "tail", 0, 1)  # legacy fixed route still requires recurrent evidence
            summary["adaptive_timing_sums"]["tail_s"] = 1
            p.write_text(json.dumps({"summary": summary, "results": rows}))
            with self.assertRaisesRegex(ValueError, "Zero draft"):
                audit_result(p, "tail", 0, 1, draft_policy=expected)

    def test_mismatched_flags_histograms_and_counters_are_rejected(self):
        for change in (lambda s, r: s["args"].update(mode="fixed"),
                       lambda s, r: s.update(draft_tokens=99),
                       lambda s, r: r[0]["adaptive"].update(cheap_policy_skips=0)):
            with self.subTest(change=change):
                summary, rows, expected = self.record()
                change(summary, rows)
                with self.assertRaises(ValueError):
                    audit_draft_policy(summary, rows, expected)


class AdaptiveShellTests(unittest.TestCase):
    setUp = cache_tests.CacheBenchmarkShellTests.setUp

    def run_shell(self, **overrides):
        env = dict(self.env)
        for key in ("DRAFT_POLICY", "TARGET_SKIP_MARGIN", "TARGET_SHORT_MARGIN", "DRAFT_MIN_MARGIN",
                    "COOLDOWN_FAILURES", "COOLDOWN_CYCLES", "SHORT_BLOCK", "MEDIUM_BLOCK", "LONG_BLOCK",
                    "MEDIUM_POSITION", "LONG_POSITION", "RUNTIME_MODE"):
            env.pop(key, None)
        env.update(overrides)
        return subprocess.run([self.bash, (ROOT / "scripts/benchmark_decode_adaptive.sh").as_posix(), "--dry-run"],
                              cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)

    def parsed(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())
        line = next(line for line in result.stdout.splitlines() if " --model-name-or-path " in line)
        return parse_decoder(shlex.split(line))

    def test_default_and_overrides_reach_real_decoder(self):
        parsed = self.parsed(self.run_shell())
        self.assertEqual((parsed.mode, parsed.cheap_verifier_margin_skip_below), ("heuristic", 1.0))
        self.assertTrue(parsed.reuse_verify_cache_for_correction)
        parsed = self.parsed(self.run_shell(BLOCK="5", TARGET_SKIP_MARGIN="0.5", TARGET_SHORT_MARGIN="2.5",
                            DRAFT_MIN_MARGIN="0.7", COOLDOWN_FAILURES="2", COOLDOWN_CYCLES="6",
                            CORRECTION_MODE="defer"))
        self.assertEqual((parsed.max_block_tokens, parsed.short_block, parsed.medium_block, parsed.long_block), (5, 5, 5, 5))
        self.assertEqual((parsed.cheap_verifier_margin_skip_below, parsed.cheap_verifier_margin_block2_below,
                          parsed.min_draft_margin, parsed.draft_cooldown_after_failures, parsed.draft_cooldown_cycles),
                         (0.5, 2.5, 0.7, 2, 6))
        self.assertTrue(parsed.defer_correction_to_next_verify)
        self.assertFalse(parsed.reuse_verify_cache_for_correction)

    def test_position_schedule_reaches_decoder_at_exact_boundaries(self):
        for policy, mode in (("schedule", "schedule"), ("adaptive", "heuristic")):
            with self.subTest(policy=policy):
                parsed = self.parsed(self.run_shell(DRAFT_POLICY=policy, BLOCK="5", SHORT_BLOCK="2",
                    MEDIUM_BLOCK="3", LONG_BLOCK="5", MEDIUM_POSITION="128", LONG_POSITION="512"))
                self.assertEqual(parsed.mode, mode)
                choose = decoder_functions()["choose_block_limit"]
                self.assertEqual([choose(parsed, position) for position in (0, 127, 128, 511, 512, 1024)],
                                 [2, 2, 3, 3, 5, 5])
                self.assertEqual(parsed.cheap_verifier_policy, "logit_margin" if policy == "adaptive" else "none")

    def test_throughput_mode_reaches_decoder_and_diagnostic_remains_default(self):
        for overrides, enabled in (({}, False), ({"RUNTIME_MODE": "throughput"}, True)):
            parsed = self.parsed(self.run_shell(**overrides))
            self.assertEqual(parsed.production_async_timing, enabled)
            self.assertEqual(parsed.compact_runtime_stats, enabled)
            self.assertEqual(parsed.draft_commit_policy, "target_match")
            self.assertEqual(parsed.target_match_lambda, 1.0)

    def test_invalid_phase_caps_and_positions_fail_before_output(self):
        for overrides in ({"LONG_BLOCK": "4"}, {"SHORT_BLOCK": "0"},
                          {"MEDIUM_POSITION": "512", "LONG_POSITION": "128"},
                          {"DRAFT_POLICY": "fixed", "SHORT_BLOCK": "1"}, {"RUNTIME_MODE": "typo"}):
            with self.subTest(overrides=overrides):
                result = self.run_shell(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_fixed_control_disables_gates_and_cooldown(self):
        parsed = self.parsed(self.run_shell(DRAFT_POLICY="fixed"))
        self.assertEqual((parsed.mode, parsed.cheap_verifier_policy), ("fixed", "none"))
        self.assertEqual((parsed.min_draft_margin, parsed.draft_cooldown_after_failures, parsed.draft_cooldown_cycles), (0, 0, 0))
        self.assertTrue(parsed.reuse_verify_cache_for_correction)

    def test_disabled_gates_and_boundary_route(self):
        p = Path(self.env["CHECKPOINT"]) / "recurft_config.json"
        p.write_text(json.dumps({"boundary_head_rank": 256, "projection_layer_ids": [29, 30, 31]}))
        parsed = self.parsed(self.run_shell(TARGET_SKIP_MARGIN="0", TARGET_SHORT_MARGIN="0", DRAFT_MIN_MARGIN="0",
                                           COOLDOWN_FAILURES="0", COOLDOWN_CYCLES="0"))
        self.assertEqual(parsed.draft_logit_source, "boundary")
        self.assertEqual((parsed.cheap_verifier_margin_skip_below, parsed.draft_cooldown_cycles), (0, 0))

    def test_bad_thresholds_and_partial_cooldown_fail_before_output(self):
        for overrides in ({"TARGET_SKIP_MARGIN": "nan"}, {"DRAFT_MIN_MARGIN": "-1"},
                          {"TARGET_SKIP_MARGIN": "4", "TARGET_SHORT_MARGIN": "3"},
                          {"COOLDOWN_FAILURES": "0"}, {"COOLDOWN_CYCLES": "0"}, {"DRAFT_POLICY": "typo"}):
            with self.subTest(overrides=overrides):
                result = self.run_shell(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
