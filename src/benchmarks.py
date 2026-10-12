"""AI Tools Benchmark Module — DevPulse AI Studio
Provides comparative benchmarks, radar metrics, and profile analysis
for ChatGPT, Google Gemini, Microsoft Copilot, and Claude 3.5 Sonnet.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AIToolsBenchmark:
    """Benchmarking and evaluation suite comparing major enterprise AI developer tools."""

    @staticmethod
    def get_benchmark_df() -> pd.DataFrame:
        """Return standardized benchmark scores across key developer dimensions."""
        data = {
            "Dimension": [
                "Python Code Synthesis (%)",
                "Debugging & Bug Hunting (%)",
                "Mathematical Reasoning (%)",
                "Context Window (k Tokens)",
                "IDE & Tooling Integration (/10)",
                "Inference Speed (Tokens/sec)"
            ],
            "OpenAI ChatGPT (GPT-4o)": [90.2, 88.5, 87.8, 128, 9.2, 85],
            "Google Gemini (1.5 Pro/Flash)": [88.7, 86.4, 89.2, 2000, 8.8, 115],
            "Microsoft Copilot": [86.5, 84.0, 81.5, 128, 9.7, 95],
            "Anthropic Claude (3.5 Sonnet)": [92.0, 91.2, 88.0, 200, 8.9, 75]
        }
        return pd.DataFrame(data)

    @staticmethod
    def get_tool_profiles() -> Dict[str, Dict[str, Any]]:
        """Return qualitative profiles, pros, cons, and best engineering use cases."""
        return {
            "Google Gemini": {
                "developer": "Google DeepMind",
                "flagship_model": "Gemini 1.5 Pro & Gemini 2.0 Flash",
                "context_window": "Up to 2,000,000 Tokens (Native Multimodal)",
                "superpower": "Massive context length allowing ingestion of entire codebases or research books at once; lightning-fast Flash inference.",
                "best_for": "Large repo architecture audits, video/audio multimodal research, high-throughput batch pipelines.",
                "pricing": "Generous free tier via Google AI Studio, highly cost-effective Flash pricing."
            },
            "OpenAI ChatGPT": {
                "developer": "OpenAI",
                "flagship_model": "GPT-4o & o1 Reasoning Series",
                "context_window": "128,000 Tokens",
                "superpower": "Pioneering reasoning capability, extensive ecosystem of Custom GPTs, Advanced Data Analysis (Code Interpreter).",
                "best_for": "Interactive exploratory coding, data visualization, conversational problem solving, and complex system design.",
                "pricing": "Tiered API pricing ($2.50 / $10.00 per 1M tokens), ChatGPT Plus subscriptions."
            },
            "Microsoft Copilot": {
                "developer": "Microsoft / GitHub",
                "flagship_model": "Copilot Chat (Powered by OpenAI & Anthropic)",
                "context_window": "Workspace RAG Indexing (~128k context)",
                "superpower": "Deepest native integration into VS Code and GitHub Pull Request reviews; inline autocomplete tab-completion.",
                "best_for": "Day-to-day inline coding flow, test generation inside the IDE, automated pull request summarization.",
                "pricing": "Subscription-based ($10–$19/month per developer)."
            },
            "Anthropic Claude": {
                "developer": "Anthropic",
                "flagship_model": "Claude 3.5 Sonnet",
                "context_window": "200,000 Tokens",
                "superpower": "Industry-leading HumanEval scores, superior nuanced code refactoring, side-by-side interactive Artifacts.",
                "best_for": "Complex front-end UI generation, intricate algorithmic refactoring, nuanced technical documentation writing.",
                "pricing": "Competitive API pricing ($3.00 / $15.00 per 1M tokens)."
            }
        }

    @classmethod
    def plot_benchmark_chart(cls, output_path: str = "charts/ai_tools_comparison.png") -> str:
        """Generate high-resolution comparative benchmark visualization."""
        df = cls.get_benchmark_df()

        # Select percentage metrics for fair normalized plotting
        plot_dimensions = ["Python Code Synthesis (%)", "Debugging & Bug Hunting (%)", "Mathematical Reasoning (%)"]
        subset = df[df["Dimension"].isin(plot_dimensions)].copy()

        tools = ["OpenAI ChatGPT (GPT-4o)", "Google Gemini (1.5 Pro/Flash)", "Microsoft Copilot", "Anthropic Claude (3.5 Sonnet)"]
        colors = ["#10a37f", "#4285f4", "#6e40c9", "#d97706"]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        x = np.arange(len(plot_dimensions))
        width = 0.18

        for idx, (tool, color) in enumerate(zip(tools, colors)):
            values = subset[tool].values
            rects = ax.bar(x + idx * width, values, width, label=tool, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
            # Add value labels on top of bars
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_ylabel('Accuracy & Benchmark Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('AI Developer Tools Benchmark: Coding, Debugging & Reasoning Performance', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([d.replace(" (%)", "") for d in plot_dimensions], fontsize=11, fontweight='bold')
        ax.set_ylim(70, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return output_path
