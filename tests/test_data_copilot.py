"""Tests for Data Copilot module."""

import unittest
import pandas as pd
import numpy as np
from src.data_copilot import DataCopilot


class TestDataCopilot(unittest.TestCase):
    def setUp(self):
        self.copilot = DataCopilot()
        # Create synthetic test dataset
        self.df = pd.DataFrame({
            "study_hours": [2.5, 5.1, 3.2, 8.5, 3.5, 15.0, 9.2, 5.5],
            "exam_score": [21.0, 47.0, 27.0, 75.0, 30.0, 95.0, 88.0, 60.0],
            "passed": ["No", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes"]
        })

    def test_inspect_dataset(self):
        inspection = self.copilot.inspect_dataset(self.df)
        self.assertEqual(inspection["num_rows"], 8)
        self.assertEqual(inspection["num_columns"], 3)
        self.assertIn("study_hours", inspection["numerical_columns"])
        self.assertIn("passed", inspection["categorical_columns"])
        self.assertEqual(inspection["total_missing_cells"], 0)

    def test_generate_eda_insights(self):
        insights = self.copilot.generate_eda_insights(self.df)
        self.assertTrue(len(insights) > 0)
        types = [i["type"] for i in insights]
        self.assertIn("Correlation", types)

    def test_recommend_ml_classification(self):
        rec = self.copilot.recommend_ml_models(self.df, "passed")
        self.assertEqual(rec["problem_type"], "Classification")
        self.assertIn("Random Forest Classifier", [m["name"] for m in rec["recommended_algorithms"]])

    def test_recommend_ml_regression(self):
        rec = self.copilot.recommend_ml_models(self.df, "exam_score")
        self.assertEqual(rec["problem_type"], "Regression")
        self.assertIn("Mean Squared Error (MSE)", rec["primary_evaluation_metrics"])

    def test_recommend_ml_invalid_column(self):
        rec = self.copilot.recommend_ml_models(self.df, "non_existent_column")
        self.assertIn("error", rec)

    def test_query_data_natural_language_summary(self):
        res = self.copilot.query_data_natural_language(self.df, "give me summary statistics")
        self.assertIn("descriptive statistics", res["interpretation"].lower())
        self.assertEqual(res["data"].shape[0], 8)  # describe() produces 8 rows: count, mean, std, etc.

    def test_query_data_natural_language_top(self):
        res = self.copilot.query_data_natural_language(self.df, "show top 3 students by exam_score")
        self.assertIn("top 3", res["interpretation"])
        self.assertEqual(len(res["data"]), 3)

    def test_query_data_natural_language_missing(self):
        res = self.copilot.query_data_natural_language(self.df, "are there any missing values?")
        self.assertIn("missing", res["interpretation"].lower())


if __name__ == "__main__":
    unittest.main()
