"""Tests for Research Copilot module."""

import unittest
from src.research_copilot import ResearchCopilot


class TestResearchCopilot(unittest.TestCase):
    def setUp(self):
        self.copilot = ResearchCopilot()
        self.sample_text = (
            "Machine learning algorithms utilize statistical optimization and gradient descent "
            "to minimize loss functions over empirical training datasets. Proper regularization, "
            "such as L1 Lasso and L2 Ridge, prevents overfitting and enhances out-of-sample generalization. "
            "In this study, Scikit-learn pipelines with 5-fold cross-validation were deployed to evaluate "
            "classification and regression performance across multiple benchmarks."
        )

    def test_summarize_executive(self):
        summary = self.copilot.summarize_text(self.sample_text, mode="executive")
        self.assertEqual(summary["mode"], "executive")
        self.assertGreater(summary["original_word_count"], 20)
        self.assertIn("Executive Abstract", summary["summary_content"])

    def test_extract_key_concepts(self):
        concepts = self.copilot.extract_key_concepts(self.sample_text)
        self.assertTrue(len(concepts) > 0)
        names = [c["concept"].lower() for c in concepts]
        self.assertTrue(any("machine learning" in n or "regularization" in n or "cross-validation" in n for n in names))

    def test_generate_study_guide(self):
        guide = self.copilot.generate_study_guide("Supervised Machine Learning")
        self.assertIn("study_guide_markdown", guide)
        self.assertIn("Supervised Machine Learning", guide["study_guide_markdown"])


if __name__ == "__main__":
    unittest.main()
