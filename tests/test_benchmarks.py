"""Tests for AI Tools Benchmark module."""

import unittest
import os
from src.benchmarks import AIToolsBenchmark


class TestAIToolsBenchmark(unittest.TestCase):
    def test_benchmark_df_structure(self):
        df = AIToolsBenchmark.get_benchmark_df()
        self.assertGreaterEqual(len(df), 5)
        self.assertIn("Dimension", df.columns)
        self.assertIn("OpenAI ChatGPT (GPT-4o)", df.columns)
        self.assertIn("Google Gemini (1.5 Pro/Flash)", df.columns)
        self.assertIn("Microsoft Copilot", df.columns)
        self.assertIn("Anthropic Claude (3.5 Sonnet)", df.columns)

    def test_tool_profiles(self):
        profiles = AIToolsBenchmark.get_tool_profiles()
        self.assertIn("Google Gemini", profiles)
        self.assertIn("OpenAI ChatGPT", profiles)
        self.assertIn("Microsoft Copilot", profiles)
        self.assertIn("Anthropic Claude", profiles)

        gemini_profile = profiles["Google Gemini"]
        self.assertIn("superpower", gemini_profile)
        self.assertIn("2,000,000", gemini_profile["context_window"])

    def test_plot_benchmark_chart(self):
        chart_dir = "charts"
        os.makedirs(chart_dir, exist_ok=True)
        out_path = os.path.join(chart_dir, "test_ai_comparison.png")
        res_path = AIToolsBenchmark.plot_benchmark_chart(out_path)
        self.assertTrue(os.path.exists(res_path))
        self.assertGreater(os.path.getsize(res_path), 1000)
        # Cleanup test chart
        if os.path.exists(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
