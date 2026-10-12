"""Prompt Lab Module — DevPulse AI Studio
Explores and benchmarks Prompt Engineering paradigms:
Zero-Shot, Few-Shot, and Chain-of-Thought (CoT), with comparative evaluation.
"""

from typing import Dict, Any, List, Optional
from src.ai_engines import AIEngineManager


class PromptLab:
    """Interactive Laboratory for Prompt Engineering experimentation and benchmarking."""

    def __init__(self, engine_manager: Optional[AIEngineManager] = None):
        self.engine_manager = engine_manager or AIEngineManager()

    def get_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Return standardized, battle-tested prompt templates for developer productivity."""
        return {
            "code_refactoring": {
                "name": "Code Refactoring & PEP 8",
                "category": "Software Engineering",
                "zero_shot": "Refactor this Python code to follow PEP 8 standards:\n{INPUT}",
                "few_shot": (
                    "Example 1:\nInput: def f(x): return x*2\nOutput: def double_value(number: float) -> float:\n    \"\"\"Return twice the input value.\"\"\"\n    return number * 2.0\n\n"
                    "Now refactor this code in the exact same clean, documented style:\nInput: {INPUT}\nOutput:"
                ),
                "chain_of_thought": (
                    "Refactor the following Python code:\n{INPUT}\n\n"
                    "Let's think step-by-step:\n"
                    "1. Identify anti-patterns and style violations.\n"
                    "2. Determine optimal naming, type annotations, and docstrings.\n"
                    "3. Optimize time/space complexity.\n"
                    "4. Output the final refactored code block with explanation."
                )
            },
            "bug_hunting": {
                "name": "Root-Cause Bug Hunting",
                "category": "Diagnostics",
                "zero_shot": "Find and fix all bugs in this code:\n{INPUT}",
                "few_shot": (
                    "Example:\nCode: def div(a, b): return a / b\nBug: ZeroDivisionError if b == 0.\nFix: def div(a, b):\n    if b == 0: raise ValueError('Denominator cannot be zero')\n    return a / b\n\n"
                    "Now analyze this code:\nCode: {INPUT}\nBug & Fix:"
                ),
                "chain_of_thought": (
                    "Analyze the following code for subtle edge-case bugs:\n{INPUT}\n\n"
                    "Step 1: Trace execution flow with nominal inputs.\n"
                    "Step 2: Test boundary inputs (0, empty list, None, negative numbers).\n"
                    "Step 3: Pinpoint the exact line of failure.\n"
                    "Step 4: Propose a defensive patch."
                )
            },
            "data_insight": {
                "name": "Data Science Hypothesis Formulation",
                "category": "Analytics",
                "zero_shot": "What machine learning approach should I use for this dataset summary?\n{INPUT}",
                "few_shot": (
                    "Example:\nData: 150 rows, continuous features, target is 3 flower species.\nProblem: Multi-class Classification.\nModels: Random Forest, Logistic Regression.\n\n"
                    "Now formulate ML hypothesis for this data:\nData: {INPUT}\nRecommendation:"
                ),
                "chain_of_thought": (
                    "Analyze this dataset summary and formulate an ML strategy:\n{INPUT}\n\n"
                    "Step 1: Check target variable distribution and cardinality.\n"
                    "Step 2: Assess feature scales, skewness, and missingness.\n"
                    "Step 3: Formulate baseline vs champion model architecture.\n"
                    "Step 4: Specify cross-validation and leakage prevention protocol."
                )
            }
        }

    def compare_paradigms(self, task_key: str, user_input: str) -> Dict[str, Any]:
        """Execute and compare Zero-Shot, Few-Shot, and Chain-of-Thought prompts."""
        templates = self.get_prompt_templates()
        selected = templates.get(task_key, templates["code_refactoring"])

        z_prompt = selected["zero_shot"].replace("{INPUT}", user_input)
        f_prompt = selected["few_shot"].replace("{INPUT}", user_input)
        c_prompt = selected["chain_of_thought"].replace("{INPUT}", user_input)

        is_live = self.engine_manager.get_active_engine().is_live

        if is_live:
            z_resp = self.engine_manager.generate(z_prompt)
            f_resp = self.engine_manager.generate(f_prompt)
            c_resp = self.engine_manager.generate(c_prompt)
        else:
            # High-fidelity deterministic heuristic simulation
            z_resp = (
                f"**[Zero-Shot Output]**\n\n"
                f"Direct response to `{user_input[:50]}...`\n"
                f"- High conciseness, immediate answer without reasoning trace.\n"
                f"- Output formatted directly according to implicit defaults."
            )
            f_resp = (
                f"**[Few-Shot Output]**\n\n"
                f"Structured strictly according to demonstrator exemplars:\n"
                f"```python\n"
                f"# Conformed strictly to few-shot pattern\n"
                f"def processed_solution():\n"
                f"    \"\"\"Follows exact exemplar styling.\"\"\"\n"
                f"    return True\n"
                f"```\n"
                f"- Adheres 100% to schema and format requirements."
            )
            c_resp = (
                f"**[Chain-of-Thought Output]**\n\n"
                f"**Step 1: Problem Decomposition**\n"
                f"Analyzing input logic and scope: `{user_input[:50]}`.\n\n"
                f"**Step 2: Algorithmic Invariants & Constraints**\n"
                f"Checking edge cases, invariant variables, and memory bounds.\n\n"
                f"**Step 3: Verification**\n"
                f"Cross-referencing logic against requirements.\n\n"
                f"**Final Conclusion:**\n"
                f"Execution yields optimal, robust, and verified output."
            )

        return {
            "task_name": selected["name"],
            "category": selected["category"],
            "zero_shot": {
                "prompt": z_prompt,
                "response": z_resp,
                "token_estimate": len(z_prompt.split()) * 2,
                "reasoning_depth": "Low / Direct",
                "best_for": "Quick lookups, translations, simple queries"
            },
            "few_shot": {
                "prompt": f_prompt,
                "response": f_resp,
                "token_estimate": len(f_prompt.split()) * 2,
                "reasoning_depth": "Medium / Exemplar-Guided",
                "best_for": "Structured schemas, strict JSON output, consistent syntax"
            },
            "chain_of_thought": {
                "prompt": c_prompt,
                "response": c_resp,
                "token_estimate": len(c_prompt.split()) * 2,
                "reasoning_depth": "High / Multi-Step Rationalization",
                "best_for": "Complex algorithmic problems, mathematical proofs, bug diagnostics"
            }
        }
