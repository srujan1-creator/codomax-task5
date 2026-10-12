"""Tests for Prompt Lab module."""

import unittest
from src.prompt_lab import PromptLab


class TestPromptLab(unittest.TestCase):
    def setUp(self):
        self.lab = PromptLab()

    def test_get_prompt_templates(self):
        templates = self.lab.get_prompt_templates()
        self.assertIn("code_refactoring", templates)
        self.assertIn("bug_hunting", templates)
        self.assertIn("data_insight", templates)

    def test_compare_paradigms(self):
        sample_code = "def add(x, y): return x + y"
        result = self.lab.compare_paradigms("code_refactoring", sample_code)

        self.assertIn("zero_shot", result)
        self.assertIn("few_shot", result)
        self.assertIn("chain_of_thought", result)

        self.assertIn("prompt", result["zero_shot"])
        self.assertIn("response", result["zero_shot"])
        self.assertIn("prompt", result["few_shot"])
        self.assertIn("prompt", result["chain_of_thought"])

        # Token estimates should be positive
        self.assertGreater(result["zero_shot"]["token_estimate"], 0)
        self.assertGreater(result["few_shot"]["token_estimate"], 0)
        self.assertGreater(result["chain_of_thought"]["token_estimate"], 0)


if __name__ == "__main__":
    unittest.main()
