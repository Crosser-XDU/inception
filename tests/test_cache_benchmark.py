"""Cache-mode command/audit checks and shell preflight without a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if sys.platform == "win32":
    sys.modules.setdefault("fcntl", ModuleType("fcntl"))
from common import audit_correction_mode
from run_decode import command


class CorrectionModeTests(unittest.TestCase):
    def args(self, mode=None):
        args = SimpleNamespace(route="tail", model=Path("base"), checkpoint=Path("ckpt"),
            data=Path("data"), output=Path("out"), start=0, samples=2, max_new_tokens=16,
            max_prompt_tokens=64, block=3, dtype="bf16", device="cuda", thinking=False)
        if mode is not None:
            args.correction_mode = mode
        return args

    def record(self, mode, count=2):
        reuse = count if mode == "reuse" else 0
        deferred = count if mode == "defer" else 0
        counters = {"fast_correction_cache_reuses": reuse, "deferred_corrective_tokens": deferred}
        summary = {"args": {"reuse_verify_cache_for_correction": mode == "reuse",
                            "defer_correction_to_next_verify": mode == "defer"}, **counters}
        return summary, [{"adaptive": dict(counters)}]

    def test_original_default_has_no_correction_switch(self):
        cmd = command(self.args())
        self.assertNotIn("--reuse-verify-cache-for-correction", cmd)
        self.assertNotIn("--defer-correction-to-next-verify", cmd)

    def test_modes_forward_exactly_one_switch_and_keep_strict_verification(self):
        for mode in ("off", "reuse", "defer"):
            with self.subTest(mode=mode):
                cmd = command(self.args(mode))
                self.assertEqual("--reuse-verify-cache-for-correction" in cmd, mode == "reuse")
                self.assertEqual("--defer-correction-to-next-verify" in cmd, mode == "defer")
                self.assertEqual(cmd[cmd.index("--draft-commit-policy") + 1], "target_match")
                self.assertEqual(cmd[cmd.index("--target-match-lambda") + 1], "1.0")
                self.assertEqual(cmd[cmd.index("--ngram-draft-mode") + 1], "off")

    def test_audit_distinguishes_enabled_from_executed(self):
        for mode in ("off", "reuse", "defer"):
            for count in (0, 2):
                with self.subTest(mode=mode, count=count):
                    summary, rows = self.record(mode, count)
                    audit = audit_correction_mode(summary, rows, mode)
                    self.assertEqual(audit["correction_mode"], mode)
                    self.assertEqual(audit["correction_optimization_observed"], mode != "off" and count > 0)

    def test_wrong_flags_fail_audit(self):
        summary, rows = self.record("off", 0)
        with self.assertRaisesRegex(ValueError, "result flag"):
            audit_correction_mode(summary, rows, "reuse")

    def test_inconsistent_counter_totals_fail_audit(self):
        summary, rows = self.record("reuse")
        rows[0]["adaptive"]["fast_correction_cache_reuses"] = 1
        with self.assertRaisesRegex(ValueError, "per-sample"):
            audit_correction_mode(summary, rows, "reuse")

    def test_missing_counters_fail_audit(self):
        summary, rows = self.record("defer")
        del rows[0]["adaptive"]["deferred_corrective_tokens"]
        with self.assertRaisesRegex(ValueError, "counter"):
            audit_correction_mode(summary, rows, "defer")

    def test_unrequested_branch_fails_audit(self):
        summary, rows = self.record("reuse")
        summary["args"]["reuse_verify_cache_for_correction"] = False
        with self.assertRaisesRegex(ValueError, "unrequested"):
            audit_correction_mode(summary, rows, "off")


class CacheBenchmarkShellTests(unittest.TestCase):
    def setUp(self):
        self.bash = shutil.which("bash")
        if os.name == "nt" and Path("D:/Git/bin/bash.exe").is_file():
            self.bash = "D:/Git/bin/bash.exe"
        if not self.bash:
            self.skipTest("Bash is required for shell preflight tests")
        self.tmp = tempfile.TemporaryDirectory(prefix="cache benchmark ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model, ckpt = self.root / "base model", self.root / "checkpoint"
        model.mkdir()
        ckpt.mkdir()
        (model / "config.json").write_text("{}")
        for name in ("adapter_config.json", "adapter_model.safetensors", "recurft_recurrent.safetensors",
                     "tokenizer_config.json", "tokenizer.json"):
            (ckpt / name).write_text("{}")
        (ckpt / "recurft_config.json").write_text(json.dumps({"boundary_head_rank": 0,
                                                            "projection_layer_ids": [29, 30, 31]}))
        (ckpt / "trainer_state.json").write_text('{"global_step":32900}')
        data = self.root / "test.jsonl"
        data.write_text("\n".join(json.dumps({"question": str(i), "answer": "1"}) for i in range(2)) + "\n")
        shim = self.root / "python startup"
        shim.mkdir()
        (shim / "sitecustomize.py").write_text(
            "import sys, types\nif sys.platform == 'win32':\n    sys.modules.setdefault('fcntl', types.ModuleType('fcntl'))\n")
        self.output = self.root / "output"
        self.env = os.environ.copy()
        for key in ("CORRECTION_MODE", "CKPT"):
            self.env.pop(key, None)
        self.env.update(MODEL=model.as_posix(), CHECKPOINT=ckpt.as_posix(), DATA=data.as_posix(),
                        GPU="0", ROUTE="auto", SAMPLES="2", START="0", MAX_NEW_TOKENS="16",
                        MAX_PROMPT_TOKENS="64", BLOCK="3", DTYPE="bf16", REPEATS="1", THINKING="0",
                        DRY_RUN="1", PYTHON=Path(sys.executable).as_posix(), PYTHONPATH=str(shim),
                        PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                        OUT=self.output.as_posix())

    def run_shell(self, mode=None, original=False):
        env = dict(self.env)
        if mode is not None:
            env["CORRECTION_MODE"] = mode
        script = "benchmark_decode.sh" if original else "benchmark_decode_cache.sh"
        return subprocess.run([self.bash, (ROOT / "scripts" / script).as_posix(), "--dry-run"],
                              cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_new_default_reaches_decoder_as_cache_reuse(self):
        result = self.run_shell()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--reuse-verify-cache-for-correction", result.stdout)
        self.assertNotIn("--defer-correction-to-next-verify", result.stdout)
        self.assertIn("--draft-logit-source tail", result.stdout)
        self.assertFalse(self.output.exists())

    def test_defer_and_off_reach_decoder_independently(self):
        for mode in ("defer", "off"):
            with self.subTest(mode=mode):
                result = self.run_shell(mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("--reuse-verify-cache-for-correction", result.stdout)
                self.assertEqual("--defer-correction-to-next-verify" in result.stdout, mode == "defer")
                self.assertFalse(self.output.exists())

    def test_original_script_keeps_optimizations_off(self):
        result = self.run_shell(original=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("--reuse-verify-cache-for-correction", result.stdout)
        self.assertNotIn("--defer-correction-to-next-verify", result.stdout)

    def test_invalid_mode_is_rejected_before_launch(self):
        result = self.run_shell("typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_boundary_auto_route_is_preserved(self):
        p = Path(self.env["CHECKPOINT"]) / "recurft_config.json"
        p.write_text(json.dumps({"boundary_head_rank": 256, "projection_layer_ids": [29, 30, 31]}))
        result = self.run_shell()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--draft-logit-source boundary", result.stdout)


if __name__ == "__main__":
    unittest.main()
