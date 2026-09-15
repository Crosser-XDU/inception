"""Regression checks for scoring, without loading the ML runtime."""

import ast
from pathlib import Path
import re
from typing import Any
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py"
NAMES = {"normalize_answer", "extract_answer", "rouge_l_f1", "score_response"}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES], type_ignores=[])
namespace = {"re": re, "Any": Any}
exec(compile(functions, str(SOURCE), "exec"), namespace)
score_response = namespace["score_response"]


class MathScoringTests(unittest.TestCase):
    def test_raw_gsm8k_rationale_uses_final_answer(self):
        row = {"answer": "Start with 1,200. Add 34 to get 1,234.\n#### 1,234"}
        self.assertEqual(score_response("Answer: 1,234", row, "math"), ("1234", "1234", 1.0))

    def test_raw_gsm8k_wrong_answer_is_rejected(self):
        row = {"answer": "Start with 9 and subtract 14.\n#### -5"}
        self.assertEqual(score_response("Answer: 9", row, "math"), ("9", "-5", 0.0))

    def test_preprocessed_references_still_work(self):
        for row in ({"answer": "3/4"}, {"answers": ["3/4"]}):
            self.assertEqual(score_response("Answer: 3/4", row, "math"), ("3/4", "3/4", 1.0))

    def test_missing_prediction_is_not_correct(self):
        self.assertEqual(score_response("No result.", {"answer": "42"}, "math")[2], 0.0)

    def test_rouge_reference_is_not_changed(self):
        row = {"answer": "A heading #### and some prose."}
        self.assertEqual(score_response(row["answer"], row, "rouge_l"), (row["answer"], row["answer"], 1.0))


if __name__ == "__main__":
    unittest.main()
