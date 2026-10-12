"""Tests for Code Copilot module."""

import unittest
from src.code_copilot import CodeCopilot


class TestCodeCopilot(unittest.TestCase):
    def setUp(self):
        self.copilot = CodeCopilot()
        self.sample_code = """
def calculate_factorial(n=5):
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

class MathUtility:
    def power(self, base, exp):
        return base ** exp
"""

    def test_ast_analysis_valid_code(self):
        res = self.copilot.analyze_ast(self.sample_code)
        self.assertTrue(res["is_valid_syntax"])
        self.assertEqual(len(res["functions"]), 2)
        self.assertEqual(len(res["classes"]), 1)
        self.assertTrue(res["has_recursion"])
        self.assertGreaterEqual(res["complexity_score"], 2)

    def test_ast_analysis_syntax_error(self):
        bad_code = "def broken_syntax(:\n    pass"
        res = self.copilot.analyze_ast(bad_code)
        self.assertFalse(res["is_valid_syntax"])
        self.assertIsNotNone(res["syntax_error"])

    def test_explain_code(self):
        explanation = self.copilot.explain_code(self.sample_code)
        self.assertIn("Code Architecture Summary", explanation)
        self.assertIn("calculate_factorial", explanation)

    def test_debug_code_detects_bugs(self):
        buggy_code = """
def faulty_func(items=[]):
    x = 10 / 0
    return items
"""
        diag = self.copilot.debug_code(buggy_code)
        self.assertGreaterEqual(diag["total_issues"], 2)
        issue_types = [i["type"] for i in diag["issues"]]
        self.assertIn("ZeroDivisionRisk", issue_types)
        self.assertIn("Code Smell", issue_types)

    def test_optimize_code(self):
        nested_code = """
def slow_matrix_mult(A, B):
    for i in range(len(A)):
        for j in range(len(B)):
            pass
"""
        opt = self.copilot.optimize_code(nested_code)
        self.assertIn("O(N^2)", opt["estimated_time_complexity"])

    def test_generate_unit_tests(self):
        tests_code = self.copilot.generate_unit_tests(self.sample_code)
        self.assertIn("unittest.TestCase", tests_code)
        self.assertIn("calculate_factorial", tests_code)

    def test_security_audit(self):
        vuln_code = """
import os
def dangerous(user_input):
    eval(user_input)
    api_key = "abcdef1234567890"
"""
        audit = self.copilot.security_audit(vuln_code)
        self.assertGreaterEqual(len(audit), 2)
        vuln_names = [a["vulnerability"] for a in audit]
        self.assertIn("Dynamic Code Execution (eval)", vuln_names)
        self.assertIn("Hardcoded Secret / Credential", vuln_names)

    def test_generate_docstrings(self):
        code_without_docs = "def multiply(x, y):\n    return x * y"
        doc_code = self.copilot.generate_docstrings(code_without_docs)
        self.assertIn('"""Execute multiply procedure.', doc_code)
        self.assertIn('Args:', doc_code)


if __name__ == "__main__":
    unittest.main()
