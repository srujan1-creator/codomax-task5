"""Main CLI Runner — DevPulse AI Studio
Codomax Internship Module 5: AI Tools & Mini Project
Usage:
    python main.py benchmark [--chart]
    python main.py code (--explain | --debug | --optimize | --tests | --audit) [--file PATH | --snippet CODE]
    python main.py data (--inspect | --query QUERY | --recommend TARGET) [--dataset PATH]
    python main.py research (--summarize TEXT | --guide TOPIC) [--mode executive|technical|action_plan]
    python main.py prompt-lab [--task code_refactoring|bug_hunting|data_insight]
    python main.py interactive
    python main.py app
"""

import sys
import os
import argparse
import pandas as pd

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ai_engines import AIEngineManager
from src.code_copilot import CodeCopilot
from src.data_copilot import DataCopilot
from src.research_copilot import ResearchCopilot
from src.prompt_lab import PromptLab
from src.benchmarks import AIToolsBenchmark


def print_banner():
    banner = r"""
================================================================================
       ____             ____        _               _    ___ 
      |  _ \  _____   _|  _ \ _   _| |___  ___     / \  |_ _|
      | | | |/ _ \ \ / / |_) | | | | / __|/ _ \   / _ \  | | 
      | |_| |  __/\ V /|  __/| |_| | \__ \  __/  / ___ \ | | 
      |____/ \___| \_/ |_|    \__,_|_|___/\___| /_/   \_\___|
         Intelligent AI Developer & Data Science Copilot Studio
                   Codomax Internship — Module 5
================================================================================
    """
    print(banner)


def get_default_dataset_path(name: str = "iris") -> str:
    candidates = {
        "iris": os.path.join("data", "iris_dataset.csv"),
        "student": os.path.join("data", "student_scores_dataset.csv"),
        "housing": os.path.join("data", "housing_dataset.csv")
    }
    for key, path in candidates.items():
        if key in name.lower() and os.path.exists(path):
            return path
    if os.path.exists(candidates["iris"]):
        return candidates["iris"]
    # Fallback search
    for fn in ["iris_dataset.csv", "student_scores_dataset.csv", "housing_dataset.csv"]:
        p = os.path.join("data", fn)
        if os.path.exists(p):
            return p
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="DevPulse AI — Developer & Data Science Copilot Studio (Module 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 1. Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Compare ChatGPT, Gemini, Copilot, and Claude")
    bench_parser.add_argument("--chart", action="store_true", help="Generate and export comparison chart")

    # 2. Code Copilot command
    code_parser = subparsers.add_parser("code", help="Code Copilot (Explain, Debug, Optimize, Tests, Audit)")
    code_group = code_parser.add_mutually_exclusive_group(required=True)
    code_group.add_argument("--explain", action="store_true", help="Explain code architecture and flow")
    code_group.add_argument("--debug", action="store_true", help="Scan and debug potential bugs")
    code_group.add_argument("--optimize", action="store_true", help="Analyze complexity and optimize")
    code_group.add_argument("--tests", action="store_true", help="Generate unit tests")
    code_group.add_argument("--audit", action="store_true", help="Security vulnerability audit")
    code_parser.add_argument("--file", type=str, help="Path to Python file to analyze")
    code_parser.add_argument("--snippet", type=str, help="Python code snippet string")

    # 3. Data Copilot command
    data_parser = subparsers.add_parser("data", help="Data Copilot (Inspect, Query, Recommend ML)")
    data_group = data_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--inspect", action="store_true", help="Inspect dataset metrics & EDA insights")
    data_group.add_argument("--query", type=str, help="Ask a natural-language query about the dataset")
    data_group.add_argument("--recommend", type=str, help="Recommend ML models for target column")
    data_parser.add_argument("--dataset", type=str, default="iris", help="Dataset name or path to CSV")

    # 4. Research Copilot command
    res_parser = subparsers.add_parser("research", help="Research Copilot (Summarize, Concepts, Study Guide)")
    res_group = res_parser.add_mutually_exclusive_group(required=True)
    res_group.add_argument("--summarize", type=str, help="Text or file path to summarize")
    res_group.add_argument("--guide", type=str, help="Topic for study guide generation")
    res_parser.add_argument("--mode", type=str, default="executive", choices=["executive", "technical", "action_plan"])

    # 5. Prompt Lab command
    prompt_parser = subparsers.add_parser("prompt-lab", help="Compare Zero-shot, Few-shot, and Chain-of-thought")
    prompt_parser.add_argument("--task", type=str, default="code_refactoring",
                               choices=["code_refactoring", "bug_hunting", "data_insight"])
    prompt_parser.add_argument("--input", type=str, default="def sum_list(vals): s = 0; for v in vals: s += v; return s")

    # 6. Interactive REPL
    subparsers.add_parser("interactive", help="Start interactive CLI terminal chat")

    # 7. Launch Web App
    subparsers.add_parser("app", help="Launch Streamlit Web Studio UI")

    args = parser.parse_args()

    if not args.command:
        print_banner()
        parser.print_help()
        print("\n💡 Quick Start Examples:")
        print("  python main.py benchmark --chart")
        print("  python main.py code --explain --snippet \"def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n\"")
        print("  python main.py data --inspect --dataset iris")
        print("  python main.py prompt-lab --task code_refactoring")
        print("  python main.py app")
        return

    engine_mgr = AIEngineManager()

    # --- Benchmark Command ---
    if args.command == "benchmark":
        print_banner()
        print("📊 AI Tools Benchmark: Comparative Performance Matrix\n")
        df = AIToolsBenchmark.get_benchmark_df()
        print(df.to_string(index=False))
        print("\n" + "="*80)
        profiles = AIToolsBenchmark.get_tool_profiles()
        for tool, p in profiles.items():
            print(f"\n🔹 {tool} ({p['developer']}):")
            print(f"   Model: {p['flagship_model']} | Context: {p['context_window']}")
            print(f"   Superpower: {p['superpower']}")
            print(f"   Best for: {p['best_for']}")

        if args.chart:
            os.makedirs("charts", exist_ok=True)
            chart_file = AIToolsBenchmark.plot_benchmark_chart("charts/ai_tools_comparison.png")
            print(f"\n✅ High-resolution benchmark chart generated: {chart_file}")

    # --- Code Command ---
    elif args.command == "code":
        code_str = ""
        if args.file and os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                code_str = f.read()
        elif args.snippet:
            code_str = args.snippet
        else:
            print("❌ Error: Must provide either --file <path> or --snippet <code>.")
            return

        copilot = CodeCopilot(engine_mgr)

        if args.explain:
            print(copilot.explain_code(code_str))
        elif args.debug:
            diag = copilot.debug_code(code_str)
            print("🔍 Diagnostic Issues Found:", diag["total_issues"])
            for iss in diag["issues"]:
                print(f"  [{iss['severity']}] {iss['type']}: {iss['message']}")
                print(f"    👉 Fix: {iss['recommendation']}")
            print("\n" + diag["ai_diagnostic"])
        elif args.optimize:
            opt = copilot.optimize_code(code_str)
            print(f"⚡ Estimated Complexity: {opt['estimated_time_complexity']}")
            print(opt["optimization_recommendations"])
        elif args.tests:
            print(copilot.generate_unit_tests(code_str))
        elif args.audit:
            audit = copilot.security_audit(code_str)
            print(f"🛡️ Security Audit: {len(audit)} vulnerability risk(s) identified.")
            for a in audit:
                print(f"  [Line {a['line']} - {a['severity']}] {a['vulnerability']}: {a['code_snippet']}")
                print(f"    👉 Recommendation: {a['recommendation']}")

    # --- Data Command ---
    elif args.command == "data":
        data_path = args.dataset
        if not os.path.exists(data_path):
            data_path = get_default_dataset_path(args.dataset)
        if not os.path.exists(data_path):
            print(f"❌ Error: Dataset file '{args.dataset}' could not be found.")
            return

        df = pd.read_csv(data_path)
        copilot = DataCopilot(engine_mgr)
        print(f"📂 Loaded dataset: {data_path} (Shape: {df.shape[0]} rows x {df.shape[1]} columns)\n")

        if args.inspect:
            metrics = copilot.inspect_dataset(df)
            print(f"• Numerical Features ({len(metrics['numerical_columns'])}): {metrics['numerical_columns']}")
            print(f"• Categorical Features ({len(metrics['categorical_columns'])}): {metrics['categorical_columns']}")
            print(f"• Total Missing Cells: {metrics['total_missing_cells']} | Duplicates: {metrics['duplicate_rows']}")
            print("\n💡 Automated EDA Insights:")
            insights = copilot.generate_eda_insights(df)
            for ins in insights:
                print(f"  [{ins['importance']}] {ins['type']}: {ins['finding']}")
        elif args.query:
            result = copilot.query_data_natural_language(df, args.query)
            print(f"🤖 Interpretation: {result['interpretation']}\n")
            print(result["data"])
            print(f"\n{result['text_response']}")
        elif args.recommend:
            rec = copilot.recommend_ml_models(df, args.recommend)
            if "error" in rec:
                print(f"❌ {rec['error']}")
            else:
                print(f"🎯 Problem Type: {rec['problem_type']}")
                print(f"📐 Validation Protocol: {rec['validation_strategy']}")
                print(f"⚙️ Preprocessing: {rec['preprocessing_pipeline']}")
                print(f"📊 Primary Metrics: {', '.join(rec['primary_evaluation_metrics'])}")
                print("\n🏆 Recommended Model Architectures:")
                for m in rec["recommended_algorithms"]:
                    print(f"  • {m['name']}: {m['rationale']}")
                print("\n" + rec["ai_architect_advice"])

    # --- Research Command ---
    elif args.command == "research":
        copilot = ResearchCopilot(engine_mgr)
        if args.summarize:
            text = args.summarize
            if os.path.exists(text):
                with open(text, "r", encoding="utf-8") as f:
                    text = f.read()
            summary = copilot.summarize_text(text, mode=args.mode)
            print(summary["summary_content"])
        elif args.guide:
            guide = copilot.generate_study_guide(args.guide)
            print(guide["study_guide_markdown"])

    # --- Prompt Lab Command ---
    elif args.command == "prompt-lab":
        print_banner()
        print(f"🔬 Prompt Engineering Lab: Benchmarking Paradigms on '{args.task}'\n")
        lab = PromptLab(engine_mgr)
        res = lab.compare_paradigms(args.task, args.input)
        print(f"Task: {res['task_name']} ({res['category']})\n")
        for paradigm in ["zero_shot", "few_shot", "chain_of_thought"]:
            data = res[paradigm]
            print("="*80)
            print(f"📌 {paradigm.replace('_', ' ').upper()} (Depth: {data['reasoning_depth']} | ~{data['token_estimate']} tokens)")
            print(f"Best For: {data['best_for']}")
            print("\nOutput:\n" + data["response"] + "\n")

    # --- Interactive REPL ---
    elif args.command == "interactive":
        print_banner()
        print("🤖 DevPulse AI Interactive Terminal REPL.")
        print("Type your questions or code tasks below. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_msg = input("DevPulse > ")
                if user_msg.strip().lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                if not user_msg.strip():
                    continue
                ans = engine_mgr.generate(user_msg)
                print(f"\n{ans}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

    # --- App Launcher ---
    elif args.command == "app":
        import subprocess
        print("🚀 Launching DevPulse AI Studio Streamlit Web Application...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
