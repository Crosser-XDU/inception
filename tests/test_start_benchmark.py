"""Start-position comparison: real CLI preflight and orchestration with synthetic decoder output."""

from collections import Counter
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest
from unittest.mock import patch

import test_adaptive_benchmark as adaptive_tests
import test_cache_benchmark as cache_tests
import benchmark_decode_start as benchmark

ROOT = Path(__file__).resolve().parents[1]


def result_for(cmd):
    a = adaptive_tests.parse_decoder(cmd)
    source = [json.loads(line) for line in Path(a.data_file).read_text().splitlines()]
    rows = []
    for sample in range(a.start_index, a.start_index + a.max_samples):
        tokens = [sample + 10, 20, 30, 40]
        position = a.latent_draft_min_position
        records = []
        cursor = 0
        while cursor < len(tokens):
            length = min(a.max_block_tokens if cursor >= position else 1, len(tokens) - cursor)
            records.append({"generated_start": cursor, "proposed_len": length, "accepted_len": length})
            cursor += length
        drafts = sum(r["proposed_len"] - 1 for r in records)
        counters = {k: 0 for k in ("cheap_policy_evaluations", "cheap_policy_skips", "draft_cooldown_skips",
                    "draft_cooldown_activations", "fast_correction_cache_reuses", "deferred_corrective_tokens",
                    "accepted_mismatch_tokens", "accepted_unchecked_tokens")}
        optimization_counts = {
            "batched_boundary_blocks": sum(r["proposed_len"] > 1 for r in records) if a.batched_draft_boundary else 0,
            "parallel_draft_steps": sum(max(0, r["proposed_len"] - 2) for r in records) if a.single_gpu_parallel_draft else 0,
            "fast_strict_blocks": len(records) if a.fast_strict_verification else 0,
            "fast_strict_draft_tokens": drafts if a.fast_strict_verification else 0}
        adaptive = {"token_ids": tokens if a.no_baseline else tokens[:-1] + [41],
            "wall_time_s": 1.5 if a.no_baseline else 1.0, "target_calls": len(records) + 1,
            "draft_tokens": drafts, "accepted_draft_tokens": drafts,
            "block_records": records, "block_hist": dict(Counter(str(r["proposed_len"]) for r in records)),
            "timings": {"t_step_s": 0.1 if drafts and not a.single_gpu_parallel_draft and a.max_block_tokens > 2 else 0,
                        "t_init_s": 0.1 if drafts else 0,
                        "draft_parallel_s": 0.1 if optimization_counts["parallel_draft_steps"] else 0,
                        "tail_s": 0.2 if drafts and a.draft_logit_source == "tail" else 0,
                        "boundary_s": 0.2 if drafts and a.draft_logit_source == "boundary" else 0},
            **counters, **optimization_counts}
        rows.append({"sample": sample, "question": source[sample]["question"], "reference_answer": "1",
            "answer_metric": "math", "prompt_tokens": 8, "generated_tokens": 4,
            "baseline_correct": None if a.no_baseline else True, "correct": sample % 2 == 0,
            "baseline": None if a.no_baseline else {"token_ids": tokens, "wall_time_s": 2.0, "target_calls": 4}, "adaptive": adaptive})
    summary = {"samples": len(rows), "args": vars(a), "ngram_draft_tokens": 0,
        "wall_clock_speedup": None if a.no_baseline else 2.0,
        "target_call_speedup": None if a.no_baseline else 1.5,
        "target_lora_merged": a.merge_target_lora,
        "recurrent_lora_merged": a.merge_recurrent_lora,
        "merged_recurrent_lora_modules": 8 if a.merge_recurrent_lora else 0,
        "adaptive_timing_sums": {k: sum(r["adaptive"]["timings"][k] for r in rows)
                                 for k in ("t_step_s", "t_init_s", "draft_parallel_s", "tail_s", "boundary_s")},
        "runtime_optimization_counts": {k: sum(r["adaptive"][k] for r in rows) for k in optimization_counts}}
    for key in ("draft_tokens", "accepted_draft_tokens", *counters):
        summary[key] = sum(r["adaptive"][key] for r in rows)
    return {"summary": summary, "results": rows}


class StartBenchmarkTests(unittest.TestCase):
    setUp = cache_tests.CacheBenchmarkShellTests.setUp

    def shell(self, *cli, **environment):
        env = dict(self.env)
        env.update(environment)
        return subprocess.run([self.bash, (ROOT / "scripts/benchmark_decode_start.sh").as_posix(), *cli],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)

    def fake_run(self, cmd, output, gpu):
        (output / "result.json").write_text(json.dumps(result_for(cmd)), encoding="utf-8")
        return {"timing_status": "sampled_exclusive"}

    def test_shell_environment_and_cli_precedence_same_data_and_checkpoint(self):
        result = self.shell("--dry-run", "--spec-start-position", "7", SPEC_START_POSITION="5",
                            DRAFT_POLICY="adaptive", TARGET_SKIP_MARGIN="999")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        cmds = [shlex.split(line.split(": ", 1)[1]) for line in result.stdout.splitlines()
                if line.startswith(("fixed: ", "after_start: "))]
        self.assertEqual(len(cmds), 2)
        fixed, later = map(adaptive_tests.parse_decoder, cmds)
        self.assertEqual((fixed.latent_draft_min_position, later.latent_draft_min_position), (0, 7))
        for a in (fixed, later):
            self.assertEqual(a.mode, "fixed")
            self.assertEqual(a.cheap_verifier_policy, "none")
            self.assertEqual(a.draft_cooldown_cycles, 0)
            self.assertEqual(a.draft_commit_policy, "target_match")
            self.assertEqual(a.target_match_lambda, 1.0)
            self.assertEqual(a.ngram_draft_mode, "off")
            self.assertTrue(a.lazy_t_sync and a.defer_t_init_until_latent)
            self.assertEqual(a.warmup_runs, 1)
            self.assertFalse(a.inplace_draft_cache or a.merge_recurrent_lora or a.anchor_hook_hidden_states)
        self.assertEqual((fixed.checkpoint, fixed.data_file, fixed.start_index, fixed.max_samples),
                         (later.checkpoint, later.data_file, later.start_index, later.max_samples))
        self.assertFalse(fixed.no_baseline)
        self.assertTrue(later.no_baseline)
        self.assertFalse(self.output.exists())

    def test_environment_only_and_start_zero(self):
        for value in ("0", "16"):
            with self.subTest(value=value):
                result = self.shell("--dry-run", SPEC_START_POSITION=value)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                line = next(s for s in result.stdout.splitlines() if s.startswith("after_start: "))
                a = adaptive_tests.parse_decoder(shlex.split(line.split(": ", 1)[1]))
                self.assertEqual(a.latent_draft_min_position, int(value))
                self.assertFalse(self.output.exists())

    def test_invalid_start_and_insufficient_data_fail_without_outputs(self):
        for cli in (("--spec-start-position", "-1"), ("--spec-start-position", "abc"), ("--start", "1")):
            with self.subTest(cli=cli):
                result = self.shell("--dry-run", *cli)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_three_way_report_shared_greedy_and_alternating_repeat_order(self):
        with patch.dict(os.environ, self.env), patch.object(benchmark, "run_logged", side_effect=self.fake_run) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main(["--no-dry-run", "--spec-start-position", "2", "--repeats", "2"])
        self.assertEqual([call.args[1].name for call in run.call_args_list], ["fixed", "after_start", "after_start", "fixed"])
        report = json.loads((self.output / "comparison.json").read_text())
        greedy, fixed, later = report["aggregate"]
        self.assertEqual([r["method"] for r in report["aggregate"]], ["greedy", "fixed", "after_start"])
        self.assertEqual(greedy["samples"], 4)  # 2 selected rows x 2 repeats, greedy counted once
        self.assertEqual((greedy["time_s"], fixed["time_s"], later["time_s"]), (8.0, 4.0, 6.0))
        self.assertAlmostEqual(later["wall_speedup"], 8 / 6)
        self.assertEqual((fixed["exact_rate"], later["exact_rate"]), (0.0, 1.0))
        self.assertEqual(later["drafting_samples"], 4)
        self.assertTrue((self.output / "comparison.csv").is_file())
        self.assertTrue((self.output / "comparison.txt").is_file())
        self.assertTrue((self.output / "complete.marker.json").is_file())

    def test_no_drafting_before_eos_still_produces_report(self):
        with patch.dict(os.environ, self.env), patch.object(benchmark, "run_logged", side_effect=self.fake_run):
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main(["--no-dry-run", "--spec-start-position", "128"])
        report = json.loads((self.output / "comparison.json").read_text())
        self.assertEqual(report["aggregate"][2]["drafting_samples"], 0)
        self.assertIsNone(report["aggregate"][2]["draft_acceptance"])
        marker = json.loads((self.output / "repeat1/after_start/complete.marker.json").read_text())
        self.assertEqual(marker["first_drafts"], [{"sample": i, "first_draft_position": None} for i in range(2)])

    def test_early_drafting_and_mismatched_rows_are_rejected(self):
        with patch.dict(os.environ, self.env):
            args = benchmark.parse_args(["--spec-start-position", "2"])
        args.route = "tail"
        fixed = result_for(benchmark.make_run(args, "fixed", self.output)[0])
        later = result_for(benchmark.make_run(args, "after_start", self.output)[0])
        benchmark.audit_start_position(later, 2)
        bad = copy.deepcopy(later)
        bad["results"][0]["adaptive"]["block_records"][-1]["generated_start"] = 1
        with self.assertRaisesRegex(ValueError, "before"):
            benchmark.audit_start_position(bad, 2)
        later["results"][0]["question"] = "different question"
        with self.assertRaisesRegex(ValueError, "sample identity"):
            benchmark.comparison(fixed, later, 2)

    def test_speed_profiles_reach_native_decoder_without_changing_target_or_acceptance(self):
        meta = Path(self.env["CHECKPOINT"]) / "recurft_config.json"
        meta.write_text(json.dumps({"boundary_head_rank": 256, "token_conditioning_rank": 0, "projection_layer_ids": [29, 30, 31]}))
        for profile, mode, inplace, merged in (("reuse", "reuse", False, False),
                ("defer", "defer", False, False), ("cache", "defer", True, False),
                ("tmerge", "defer", True, True), ("batch", "defer", True, True),
                ("parallel", "defer", True, True), ("strict", "defer", True, True),
                ("batch_strict", "defer", True, True), ("parallel_strict", "defer", True, True),
                ("target_merge", "defer", True, True), ("target_hook", "defer", True, True)):
            with self.subTest(profile=profile):
                env = dict(self.env, SPEED_PROFILE=profile)
                result = subprocess.run([self.bash, (ROOT / "scripts/benchmark_decode_speed.sh").as_posix(),
                    "--dry-run"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                lines = [line for line in result.stdout.splitlines() if line.startswith(("fixed: ", "after_start: "))]
                self.assertEqual(len(lines), 2)
                for line in lines:
                    a = adaptive_tests.parse_decoder(shlex.split(line.split(": ", 1)[1]))
                    self.assertEqual(a.reuse_verify_cache_for_correction, mode == "reuse")
                    self.assertEqual(a.defer_correction_to_next_verify, mode == "defer")
                    self.assertEqual(a.inplace_draft_cache, inplace)
                    self.assertEqual(a.merge_recurrent_lora, merged)
                    self.assertEqual(a.batched_draft_boundary, profile in {"batch", "batch_strict", "target_merge", "target_hook"})
                    self.assertEqual(a.single_gpu_parallel_draft, profile in {"parallel", "parallel_strict"})
                    self.assertEqual(a.fast_strict_verification, "strict" in profile or profile in {"target_merge", "target_hook"})
                    self.assertEqual(a.merge_target_lora, profile in {"target_merge", "target_hook"})
                    self.assertEqual(a.anchor_hook_hidden_states, profile == "target_hook")
                    self.assertFalse(a.precommit_unchecked_drafts)
                    self.assertEqual((a.draft_commit_policy, a.target_match_lambda), ("target_match", 1.0))
                self.assertFalse(self.output.exists())

    def test_runtime_flags_override_environment_and_reject_unchecked_combination(self):
        result = self.shell("--dry-run", "--no-inplace-draft-cache", "--merge-recurrent-lora",
                            "--anchor-hook-hidden-states", INPLACE_DRAFT_CACHE="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("fixed: "))
        a = adaptive_tests.parse_decoder(shlex.split(line.split(": ", 1)[1]))
        self.assertFalse(a.inplace_draft_cache)
        self.assertTrue(a.merge_recurrent_lora and a.anchor_hook_hidden_states)
        for cli, env in ((("--correction-mode", "off"), {"INPLACE_DRAFT_CACHE": "1"}),
                         ((), {"MERGE_RECURRENT_LORA": "yes"})):
            result = self.shell("--dry-run", *cli, **env)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.output.exists())

    def test_rejection_and_correction_statistics_survive_repeat_aggregation(self):
        with patch.dict(os.environ, self.env):
            args = benchmark.parse_args(["--spec-start-position", "2"])
        args.route = "tail"
        fixed = result_for(benchmark.make_run(args, "fixed", self.output)[0])
        later = result_for(benchmark.make_run(args, "after_start", self.output)[0])
        for row, accepted in zip(fixed["results"], (0, 1)):
            row["adaptive"]["accepted_draft_tokens"] = accepted
            row["adaptive"]["block_records"][0]["accepted_len"] = 1 + accepted
            row["adaptive"]["deferred_corrective_tokens"] = 1
        metrics = benchmark.comparison(fixed, later, 2)
        rows = benchmark.aggregate([{"metrics": metrics}, {"metrics": metrics}])
        self.assertEqual(rows[1]["draft_blocks"], 4)
        self.assertEqual(rows[1]["mean_accepted_drafts"], 0.5)
        self.assertEqual(rows[1]["zero_accept_rate"], 0.5)
        self.assertEqual(rows[1]["full_accept_rate"], 0)
        self.assertEqual(rows[1]["deferred_corrections"], 4)
        self.assertIsNone(rows[0]["mean_accepted_drafts"])

    def test_merge_must_be_observed_and_runtime_flags_match_across_runs(self):
        def failed_merge(cmd, output, gpu):
            data = result_for(cmd)
            data["summary"]["merged_recurrent_lora_modules"] = 0
            (output / "result.json").write_text(json.dumps(data), encoding="utf-8")
            return {"timing_status": "sampled_exclusive"}
        with patch.dict(os.environ, self.env), patch.object(benchmark, "run_logged", side_effect=failed_merge):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "no merged modules"):
                benchmark.main(["--no-dry-run", "--merge-recurrent-lora"])
        self.assertTrue((self.output / "failed.marker.json").exists())
        with patch.dict(os.environ, self.env):
            args = benchmark.parse_args([])
        args.route = "tail"
        fixed = result_for(benchmark.make_run(args, "fixed", self.output)[0])
        later = result_for(benchmark.make_run(args, "after_start", self.output)[0])
        later["summary"]["args"]["inplace_draft_cache"] = True
        with self.assertRaisesRegex(ValueError, "shared experiment settings"):
            benchmark.comparison(fixed, later, 128)

    def test_runtime_execution_coverage_and_parallel_timing(self):
        meta = Path(self.env["CHECKPOINT"]) / "recurft_config.json"
        meta.write_text(json.dumps({"boundary_head_rank": 256, "token_conditioning_rank": 0, "projection_layer_ids": [29, 30, 31]}))
        for flags in (("--batched-draft-boundary",), ("--single-gpu-parallel-draft",),
                      ("--fast-strict-verification",), ("--batched-draft-boundary", "--fast-strict-verification")):
            with self.subTest(flags=flags), patch.dict(os.environ, self.env):
                args = benchmark.parse_args(["--route", "boundary", *flags])
                cmd, _ = benchmark.make_run(args, "fixed", self.output)
                data = result_for(cmd)
                totals = benchmark.audit_runtime_optimizations(data)
                self.assertEqual(totals, data["summary"]["runtime_optimization_counts"])
                key = ("batched_boundary_blocks" if args.batched_draft_boundary else
                       "parallel_draft_steps" if args.single_gpu_parallel_draft else "fast_strict_blocks")
                self.assertGreater(totals[key], 0)
                bad = copy.deepcopy(data)
                bad["results"][0]["adaptive"][key] = 0
                with self.assertRaises(ValueError):
                    benchmark.audit_runtime_optimizations(bad)
        with patch.dict(os.environ, self.env), patch.object(benchmark, "run_logged", side_effect=self.fake_run):
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main(["--no-dry-run", "--route", "boundary", "--single-gpu-parallel-draft",
                                "--fast-strict-verification"])
        report = json.loads((self.output / "comparison.json").read_text())
        self.assertGreater(report["aggregate"][1]["parallel_draft_steps"], 0)
        self.assertEqual(report["aggregate"][1]["fast_strict_draft_tokens"], report["aggregate"][1]["draft_tokens"])

    def test_incompatible_draft_modes_and_conditioned_checkpoints_fail_preflight(self):
        result = self.shell("--dry-run", "--batched-draft-boundary", "--single-gpu-parallel-draft")
        self.assertNotEqual(result.returncode, 0)
        meta = Path(self.env["CHECKPOINT"]) / "recurft_config.json"
        meta.write_text(json.dumps({"boundary_head_rank": 256, "token_conditioning_rank": 8, "projection_layer_ids": [29, 30, 31]}))
        result = self.shell("--dry-run", "--batched-draft-boundary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("token conditioning", result.stderr)
        self.assertFalse(self.output.exists())

    def test_fast_strict_diagnostic_timing_keeps_compact_stats(self):
        result = self.shell("--dry-run", "--fast-strict-verification", "--runtime-mode", "diagnostic")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("fixed: "))
        a = adaptive_tests.parse_decoder(shlex.split(line.split(": ", 1)[1]))
        self.assertTrue(a.fast_strict_verification and a.compact_runtime_stats)
        self.assertFalse(a.production_async_timing)

    def test_changed_checkpoint_aborts_before_next_run(self):
        def mutate(cmd, output, gpu):
            status = self.fake_run(cmd, output, gpu)
            (Path(self.env["CHECKPOINT"]) / "adapter_model.safetensors").write_text("changed")
            return status
        with patch.dict(os.environ, self.env), patch.object(benchmark, "run_logged", side_effect=mutate) as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "changed"):
                benchmark.main(["--no-dry-run", "--spec-start-position", "2"])
        self.assertEqual(run.call_count, 1)
        self.assertTrue((self.output / "failed.marker.json").is_file())
        self.assertFalse((self.output / "complete.marker.json").exists())


if __name__ == "__main__":
    unittest.main()
