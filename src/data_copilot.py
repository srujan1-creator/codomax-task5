"""Data Copilot Module — DevPulse AI Studio
Enhances and unifies previous Module 3 & 4 projects with AI-driven dataset
diagnostics, natural-language query resolution, EDA insights, and ML model recommendation.
"""

import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.ai_engines import AIEngineManager


class DataCopilot:
    """Intelligent AI Data Science & Machine Learning Copilot."""

    def __init__(self, engine_manager: Optional[AIEngineManager] = None):
        self.engine_manager = engine_manager or AIEngineManager()

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical and structural metrics for a pandas DataFrame."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2).to_dict()

        # Skewness
        skewness_dict = {}
        if num_cols:
            skewness = df[num_cols].skew().round(3).to_dict()
            skewness_dict = skewness

        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "missing_percentages": missing_pct,
            "total_missing_cells": int(missing_counts.sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
            "skewness": skewness_dict
        }

    def generate_eda_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate automated domain insights from distributions, correlations, and outliers."""
        insights = []
        metrics = self.inspect_dataset(df)

        # 1. Missingness & Duplicates
        if metrics["total_missing_cells"] == 0:
            insights.append({
                "type": "Data Quality",
                "finding": "Perfect data completeness: 0 null or NaN cells detected across all records.",
                "importance": "POSITIVE"
            })
        else:
            high_miss = [col for col, pct in metrics["missing_percentages"].items() if pct > 5.0]
            insights.append({
                "type": "Data Quality",
                "finding": f"Missing values present: {metrics['total_missing_cells']} total cells. Columns needing imputation: {high_miss or 'low <5%'}.",
                "importance": "WARNING"
            })

        if metrics["duplicate_rows"] > 0:
            insights.append({
                "type": "Data Quality",
                "finding": f"Detected {metrics['duplicate_rows']} duplicate row(s) that should be deduplicated prior to training.",
                "importance": "WARNING"
            })

        # 2. Correlation Insights
        num_cols = metrics["numerical_columns"]
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            strong_pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_matrix.loc[c1, c2]
                    if abs(val) >= 0.70:
                        strong_pairs.append((c1, c2, round(val, 3)))

            if strong_pairs:
                for c1, c2, r in strong_pairs:
                    insights.append({
                        "type": "Correlation",
                        "finding": f"Strong correlation between '{c1}' and '{c2}' (Pearson r = {r}). Strong predictive signal or collinearity.",
                        "importance": "KEY_INSIGHT"
                    })
            else:
                insights.append({
                    "type": "Correlation",
                    "finding": "Features demonstrate mild or independent correlations (all |r| < 0.70), minimizing collinearity risks.",
                    "importance": "INFO"
                })

        # 3. Outlier Detection via IQR
        outlier_cols = []
        for col in num_cols:
            series = df[col].dropna()
            if len(series) > 10:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    num_outliers = int(((series < lower) | (series > upper)).sum())
                    if num_outliers > 0:
                        pct_out = round(num_outliers / len(series) * 100, 1)
                        if pct_out > 2.0:
                            outlier_cols.append(f"{col} ({num_outliers} records, {pct_out}%)")

        if outlier_cols:
            insights.append({
                "type": "Outlier Detection",
                "finding": f"Statistical outliers detected via 1.5x IQR rule in: {', '.join(outlier_cols)}.",
                "importance": "INFO"
            })

        # 4. Target / Class distribution
        cat_cols = metrics["categorical_columns"]
        for col in cat_cols:
            cardinality = df[col].nunique()
            if 2 <= cardinality <= 10:
                val_counts = df[col].value_counts(normalize=True).to_dict()
                min_ratio = min(val_counts.values())
                max_ratio = max(val_counts.values())
                balance_desc = "Well balanced" if min_ratio / max_ratio > 0.6 else "Mildly or strongly imbalanced"
                insights.append({
                    "type": "Distribution",
                    "finding": f"Category '{col}' contains {cardinality} distinct classes. Balance status: {balance_desc}.",
                    "importance": "INFO"
                })

        return insights

    def recommend_ml_models(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze dataset target column and recommend optimal ML algorithms and preprocessing."""
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' does not exist in dataset."}

        target_series = df[target_column].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        is_float = pd.api.types.is_float_dtype(target_series)
        num_unique = target_series.nunique()

        # Problem Type Determination:
        # Categorical, boolean, or low-cardinality integers are classification.
        # Floating point or high-cardinality numeric variables are regression.
        if not is_numeric or (not is_float and num_unique <= 10):
            problem_type = "Classification"
            recommended_models = [
                {"name": "Random Forest Classifier", "rationale": "High accuracy, handles non-linear relationships, robust against feature scale variance."},
                {"name": "Logistic Regression (L2 / Ridge)", "rationale": "Fast baseline with probabilistic outputs, interpretable log-odds coefficients."},
                {"name": "Support Vector Machine (RBF Kernel)", "rationale": "Effective in high-dimensional separable spaces."},
                {"name": "K-Nearest Neighbors (KNN)", "rationale": "Non-parametric distance-based clustering, intuitive local boundaries."}
            ]
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            split_strategy = "StratifiedKFold (5 Folds) with Stratified Train-Test Split (80/20)"
            scaling = "StandardScaler (mean=0, std=1) fit strictly on train split to prevent data leakage."
        else:
            problem_type = "Regression"
            recommended_models = [
                {"name": "Ridge Regression (L2)", "rationale": "Mitigates multicollinearity via weight penalty, robust continuous baseline."},
                {"name": "Random Forest Regressor", "rationale": "Captures non-linear feature interactions without strict linearity assumptions."},
                {"name": "Lasso Regression (L1)", "rationale": "Encourages sparse weights, acting as an automated feature selector."},
                {"name": "Gradient Boosting Regressor", "rationale": "Sequential error correction achieving superior residual minimization."}
            ]
            metrics = ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R-squared (R²)"]
            split_strategy = "Standard K-Fold (5 Folds) with Train-Test Split (80/20)"
            scaling = "StandardScaler or RobustScaler applied on numerical predictor features."

        prompt = (
            f"Given a dataset with {len(df)} rows, target column '{target_column}' ({num_unique} unique values, {target_series.dtype}), "
            f"the problem type is {problem_type}. Provide concise engineering recommendations for feature engineering and tuning."
        )
        ai_recommendation = self.engine_manager.generate(prompt, "You are a Lead Data Science & ML Architect.")

        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "target_unique_values": num_unique,
            "recommended_algorithms": recommended_models,
            "primary_evaluation_metrics": metrics,
            "validation_strategy": split_strategy,
            "preprocessing_pipeline": scaling,
            "ai_architect_advice": ai_recommendation
        }

    def query_data_natural_language(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Translate natural language questions into real Pandas operations and return results."""
        q_clean = query.strip().lower()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. Summary stats / describe
        if "summary" in q_clean or "describe" in q_clean or "statistics" in q_clean:
            stats = df.describe().round(2)
            return {
                "interpretation": "Computed summary descriptive statistics across all numerical columns.",
                "data": stats,
                "text_response": f"Descriptive statistics calculated for {len(num_cols)} numerical features across {len(df)} rows."
            }

        # 2. Missing values
        if "missing" in q_clean or "null" in q_clean or "nan" in q_clean:
            miss = df.isnull().sum().reset_index()
            miss.columns = ["Column", "Missing_Count"]
            miss["Percentage"] = (miss["Missing_Count"] / len(df) * 100).round(2)
            total_miss = miss["Missing_Count"].sum()
            return {
                "interpretation": "Identified null/missing values per column.",
                "data": miss,
                "text_response": f"Total missing cells across dataframe: {total_miss}."
            }

        # 3. Correlation
        if "correlation" in q_clean or "correlate" in q_clean:
            if num_cols:
                corr = df[num_cols].corr().round(3)
                return {
                    "interpretation": "Computed Pearson correlation matrix across numerical variables.",
                    "data": corr,
                    "text_response": f"Correlation matrix generated for features: {', '.join(num_cols)}."
                }

        # 4. Top / Bottom N rows sorted by column
        match_top = re.search(r"(top|highest|best|first)\s+(\d+)?", q_clean)
        if match_top:
            n = int(match_top.group(2)) if match_top.group(2) else 5
            # Find best candidate numeric column mentioned or last numeric column
            target_col = None
            for col in num_cols:
                if col.lower() in q_clean:
                    target_col = col
                    break
            if not target_col and num_cols:
                target_col = num_cols[-1]

            if target_col:
                subset = df.sort_values(by=target_col, ascending=False).head(n)
                return {
                    "interpretation": f"Filtered top {n} records sorted descending by '{target_col}'.",
                    "data": subset,
                    "text_response": f"Retrieved top {n} entries ordered by highest {target_col}."
                }

        # 5. Average / Mean by category
        if "average" in q_clean or "mean" in q_clean:
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if cat_cols and num_cols:
                grp_col = cat_cols[0]
                val_col = num_cols[0]
                for c in cat_cols:
                    if c.lower() in q_clean:
                        grp_col = c
                        break
                for n_c in num_cols:
                    if n_c.lower() in q_clean:
                        val_col = n_c
                        break

                grouped = df.groupby(grp_col)[val_col].agg(["mean", "std", "count"]).round(2).reset_index()
                return {
                    "interpretation": f"Calculated mean and std of '{val_col}' grouped by '{grp_col}'.",
                    "data": grouped,
                    "text_response": f"Group-wise aggregation computed for '{val_col}' across '{grp_col}' categories."
                }

        # Default fallback: preview first 5 rows
        sample = df.head(5)
        return {
            "interpretation": f"Displaying nominal preview slice matching query: '{query}'.",
            "data": sample,
            "text_response": f"Showing first 5 records from dataset of shape {df.shape}."
        }
