import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import audit_result, validate_data
from run_decode import command


class LauncherTests(unittest.TestCase):
    def test_command_disables_ngram_for_neural_routes(self):
        for route in ("tail", "boundary"):
            args = SimpleNamespace(route=route, model=Path("base"), checkpoint=Path("ckpt"),
                data=Path("data"), output=Path("out"), start=8, samples=384, max_new_tokens=256,
                max_prompt_tokens=1024, block=3, dtype="fp16", device="cuda", thinking=False)
            cmd = command(args)
            self.assertEqual(cmd[cmd.index("--ngram-draft-mode") + 1], "off")
            self.assertEqual(cmd[cmd.index("--draft-logit-source") + 1], route)

    def test_short_or_duplicate_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text(json.dumps({"question": "q", "answer": "a"}) + '\n')
            with self.assertRaises(ValueError):
                validate_data(p, 0, 32)
            p.write_text(p.read_text() * 2)
            with self.assertRaises(ValueError):
                validate_data(p, 0, 2)

    def test_blank_line_cannot_shift_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text('\n' + json.dumps({"question": "q", "answer": "a"}) + '\n')
            with self.assertRaises(ValueError):
                validate_data(p, 0, 1)

    def test_ngram_only_cannot_pass_recurrent_audit(self):
        record = {"summary": {"samples": 1, "args": {"draft_commit_policy": "target_match",
                  "ngram_draft_mode": "only", "draft_logit_source": "boundary", "target_match_lambda": 1.0},
                  "accepted_mismatch_tokens": 0, "accepted_unchecked_tokens": 0,
                  "adaptive_timing_sums": {"t_step_s": 0, "boundary_s": 0, "tail_s": 0},
                  "draft_tokens": 10, "ngram_draft_tokens": 10},
                  "results": [{"sample": 0, "adaptive": {"accepted_mismatch_tokens": 0,
                                                         "accepted_unchecked_tokens": 0}}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                audit_result(p, "boundary", 0, 1)

    def test_missing_safety_counters_are_rejected(self):
        record = {"summary": {"samples": 1, "args": {"draft_commit_policy": "target_match",
                  "target_match_lambda": 1.0}, "adaptive_timing_sums": {}},
                  "results": [{"sample": 0, "adaptive": {}}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            p.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Missing safety counter"):
                audit_result(p, "boundary", 0, 1)

    def test_relaxed_threshold_is_rejected(self):
        record = {"summary": {"samples": 1, "args": {"draft_commit_policy": "target_match",
                  "target_match_lambda": 0.9}, "adaptive_timing_sums": {}},
                  "results": [{"sample": 0, "adaptive": {}}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            p.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "strict target-match"):
                audit_result(p, "boundary", 0, 1)
            record["summary"]["args"].update(target_match_lambda=1.0, target_match_lambda_late=0.9)
            p.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "position-specific"):
                audit_result(p, "boundary", 0, 1)

    def test_nonfinite_and_wrong_coverage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            p.write_text('{"value": NaN}')
            with self.assertRaises(ValueError):
                audit_result(p, "tail", 0, 1)
            p.write_text(json.dumps({"summary": {"samples": 1}, "results": [{"sample": 2}]}))
            with self.assertRaises(ValueError):
                audit_result(p, "tail", 0, 1)


if __name__ == "__main__":
    unittest.main()
