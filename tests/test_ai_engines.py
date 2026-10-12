"""Tests for AI Engines module."""

import unittest
from src.ai_engines import SmartOfflineEngine, AIEngineManager, LiveGeminiEngine, LiveOpenAIEngine


class TestAIEngines(unittest.TestCase):
    def setUp(self):
        self.offline = SmartOfflineEngine()
        self.manager = AIEngineManager(default_engine="offline")

    def test_smart_offline_code_explanation(self):
        resp = self.offline.generate("Explain this code: def add(a, b): return a + b")
        self.assertIn("DevPulse AI", resp)
        self.assertIn("Explanation", resp)

    def test_smart_offline_debug(self):
        resp = self.offline.generate("Debug this error in python script")
        self.assertIn("Debug", resp)

    def test_smart_offline_optimize(self):
        resp = self.offline.generate("Optimize complexity of nested loop")
        self.assertIn("Optimization", resp)

    def test_smart_offline_unit_tests(self):
        resp = self.offline.generate("Generate unit tests for my function")
        self.assertIn("unittest", resp)

    def test_smart_offline_data_ml(self):
        resp = self.offline.generate("Analyze this dataset correlation for machine learning")
        self.assertIn("Data Science", resp)

    def test_smart_offline_research(self):
        resp = self.offline.generate("Summarize this technical abstract")
        self.assertIn("Research Synthesis", resp)

    def test_smart_offline_prompt_lab(self):
        resp = self.offline.generate("Compare zero-shot vs few-shot prompt")
        self.assertIn("Prompt Engineering", resp)

    def test_engine_manager_list_engines(self):
        engines = self.manager.list_engines()
        self.assertEqual(len(engines), 3)
        engine_ids = [e["id"] for e in engines]
        self.assertIn("offline", engine_ids)
        self.assertIn("gemini", engine_ids)
        self.assertIn("openai", engine_ids)

    def test_engine_manager_fallback(self):
        # Even if set to gemini without valid key, safely falls back
        self.manager.set_active_engine("gemini")
        resp = self.manager.generate("Hello world")
        self.assertTrue(len(resp) > 0)


if __name__ == "__main__":
    unittest.main()
