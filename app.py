"""DevPulse AI — Intelligent AI Developer & Data Copilot Studio
Streamlit Web Application
Codomax Internship — Module 5 (Day 17 - Day 20)
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure local src imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ai_engines import AIEngineManager, SmartOfflineEngine
from src.code_copilot import CodeCopilot
from src.data_copilot import DataCopilot
from src.research_copilot import ResearchCopilot
from src.prompt_lab import PromptLab
from src.benchmarks import AIToolsBenchmark

# --- Page Configuration ---
st.set_page_config(
    page_title="DevPulse AI — Developer & Data Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-blue { background-color: #dbeafe; color: #1e40af; }
    .badge-green { background-color: #dcfce7; color: #166534; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "engine_manager" not in st.session_state:
    st.session_state.engine_manager = AIEngineManager(default_engine="auto")

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.shields.io/badge/Codomax-Module%205%20(Day%2017--20)-blue?style=for-the-badge", use_container_width=True)
    st.title("⚙️ Engine Settings")

    engine_choice = st.selectbox(
        "Select AI Inference Engine:",
        options=["Smart Offline Engine (Zero-Config)", "Google Gemini (Live API)", "OpenAI (Live API)"],
        index=0
    )

    api_key_input = None
    if "Gemini" in engine_choice:
        api_key_input = st.text_input("Enter Gemini API Key (optional):", type="password", help="Leave blank to use .env or offline fallback.")
        st.session_state.engine_manager.set_active_engine("gemini", key=api_key_input if api_key_input else None)
    elif "OpenAI" in engine_choice:
        api_key_input = st.text_input("Enter OpenAI API Key (optional):", type="password", help="Leave blank to use .env or offline fallback.")
        st.session_state.engine_manager.set_active_engine("openai", key=api_key_input if api_key_input else None)
    else:
        st.session_state.engine_manager.set_active_engine("offline")

    active_eng = st.session_state.engine_manager.get_active_engine()
    st.markdown(f"**Status:** `🟢 {active_eng.name}`")
    st.markdown(f"**Mode:** {'⚡ Live API' if active_eng.is_live else '🔒 Offline Deterministic'}")

    st.markdown("---")
    st.markdown("### 📌 Navigation")
    st.markdown("""
    - **Tab 1:** Code Copilot & Reviewer
    - **Tab 2:** Data Science & ML Copilot
    - **Tab 3:** Research & Synthesizer
    - **Tab 4:** Prompt Engineering Lab
    - **Tab 5:** AI Tools Benchmark
    """)

    st.markdown("---")
    st.markdown("**Author:** [srujan1-creator](https://github.com/srujan1-creator)")
    st.markdown("**Internship:** Codomax Python AI/ML Track")

# --- Main App Header ---
st.markdown('<div class="main-title">DevPulse AI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent AI Developer, Data Science & Research Copilot | Codomax Internship Module 5</div>', unsafe_allow_html=True)

# --- App Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💻 Code Copilot & Reviewer",
    "📊 Data & ML Copilot",
    "📑 Research Synthesizer",
    "🔬 Prompt Engineering Lab",
    "⚡ AI Tools Benchmark"
])

# ==============================================================================
# TAB 1: CODE COPILOT
# ==============================================================================
with tab1:
    st.header("💻 AI Code Copilot & Reviewer")
    st.caption("Perform automated AST inspection, syntax debugging, algorithmic complexity analysis, unit test generation, and security auditing.")

    code_copilot = CodeCopilot(st.session_state.engine_manager)

    sample_presets = {
        "Recursive Fibonacci (Complexity Risk)": (
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        ),
        "Buggy List Processor (Anti-Patterns)": (
            "def append_to_list(element, target=[]):\n"
            "    target.append(element)\n"
            "    ratio = element / 0\n"
            "    return target\n"
        ),
        "Vulnerable Shell Runner (Security Risk)": (
            "import subprocess\n"
            "def run_command(user_query):\n"
            "    api_key = 'sk_live_998877665544332211'\n"
            "    eval(user_query)\n"
            "    subprocess.run(f'echo {user_query}', shell=True)\n"
        ),
        "Custom Code Snippet": ""
    }

    selected_preset = st.selectbox("Select Code Example or Custom Snippet:", list(sample_presets.keys()))
    default_code = sample_presets[selected_preset] if sample_presets[selected_preset] else "def example():\n    pass"

    code_input = st.text_area("Python Code Editor:", value=default_code, height=180)

    # Action Buttons Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    btn_explain = c1.button("🧠 Explain", use_container_width=True)
    btn_debug = c2.button("🔍 Debug", use_container_width=True)
    btn_optimize = c3.button("⚡ Optimize", use_container_width=True)
    btn_tests = c4.button("🧪 Unit Tests", use_container_width=True)
    btn_audit = c5.button("🛡️ Security Audit", use_container_width=True)
    btn_docstrings = c6.button("📝 Add Docs", use_container_width=True)

    # AST Metrics Banner
    if code_input.strip():
        ast_metrics = code_copilot.analyze_ast(code_input)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Syntax", "Valid" if ast_metrics["is_valid_syntax"] else "Error")
        m1.caption("Python 3.10+ Spec")
        m2.metric("Lines of Code", ast_metrics["lines_of_code"])
        m3.metric("Functions", len(ast_metrics["functions"]))
        m4.metric("Classes", len(ast_metrics["classes"]))
        m5.metric("Complexity Score", ast_metrics["complexity_score"])

    st.markdown("---")

    if btn_explain:
        st.subheader("📋 Architectural & Code Explanation")
        result = code_copilot.explain_code(code_input)
        st.markdown(result)

    elif btn_debug:
        st.subheader("🔍 Automated Diagnostics & Bug Detection")
        diag = code_copilot.debug_code(code_input)
        if diag["total_issues"] == 0:
            st.success("✅ No critical syntax errors or common anti-patterns detected!")
        else:
            st.warning(f"⚠️ Found {diag['total_issues']} potential issue(s):")
            for iss in diag["issues"]:
                st.error(f"**[{iss['severity']}] {iss['type']}**: {iss['message']}\n\n*Recommendation:* {iss['recommendation']}")
        st.markdown("---")
        st.markdown(diag["ai_diagnostic"])

    elif btn_optimize:
        st.subheader("⚡ Algorithmic Optimization & Complexity Analysis")
        opt = code_copilot.optimize_code(code_input)
        st.info(f"**Estimated Time Complexity:** `{opt['estimated_time_complexity']}`")
        st.markdown(opt["optimization_recommendations"])

    elif btn_tests:
        st.subheader("🧪 Automated Unit Test Generation")
        tests_code = code_copilot.generate_unit_tests(code_input)
        st.code(tests_code.replace("```python\n", "").replace("```", ""), language="python")
        st.download_button("📥 Download Test Suite", tests_code, file_name="test_generated_suite.py")

    elif btn_audit:
        st.subheader("🛡️ Security Vulnerability Audit")
        findings = code_copilot.security_audit(code_input)
        if not findings:
            st.success("✅ Security Audit Passed: No known critical vulnerabilities (eval, injection, hardcoded secrets) found.")
        else:
            for f in findings:
                st.error(f"**Line {f['line']} [{f['severity']}]: {f['vulnerability']}**\n\n`{f['code_snippet']}`\n\n*Fix:* {f['recommendation']}")

    elif btn_docstrings:
        st.subheader("📝 Auto-Generated Docstrings")
        styled_code = code_copilot.generate_docstrings(code_input)
        st.code(styled_code, language="python")

# ==============================================================================
# TAB 2: DATA & ML COPILOT
# ==============================================================================
with tab2:
    st.header("📊 AI Data Science & Machine Learning Copilot")
    st.caption("Enhances Module 3 & 4 with natural-language query resolution, automated EDA insight generation, and ML model architecture recommendations.")

    data_copilot = DataCopilot(st.session_state.engine_manager)

    dataset_choice = st.radio(
        "Choose Data Source:",
        ["Iris Botanical Flowers (Classification)", "Student Exam Performance (Regression)", "Upload Custom CSV"],
        horizontal=True
    )

    df_active = None
    if "Iris" in dataset_choice:
        p = os.path.join("data", "iris_dataset.csv")
        df_active = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    elif "Student" in dataset_choice:
        p = os.path.join("data", "student_scores_dataset.csv")
        df_active = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    else:
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        if uploaded_file is not None:
            df_active = pd.read_csv(uploaded_file)

    if df_active is not None and not df_active.empty:
        # Metrics Row
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Rows", df_active.shape[0])
        d2.metric("Total Columns", df_active.shape[1])
        d3.metric("Missing Cells", int(df_active.isnull().sum().sum()))
        d4.metric("Duplicate Rows", int(df_active.duplicated().sum()))

        with st.expander("🔍 Dataset Preview & Schema", expanded=True):
            st.dataframe(df_active.head(6), use_container_width=True)

        # Action Sections
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            st.subheader("💡 Automated EDA Insights")
            if st.button("Generate Statistical Insights"):
                insights = data_copilot.generate_eda_insights(df_active)
                for ins in insights:
                    if ins["importance"] == "POSITIVE":
                        st.success(f"**{ins['type']}**: {ins['finding']}")
                    elif ins["importance"] == "KEY_INSIGHT":
                        st.info(f"**{ins['type']}**: {ins['finding']}")
                    elif ins["importance"] == "WARNING":
                        st.warning(f"**{ins['type']}**: {ins['finding']}")
                    else:
                        st.write(f"• **{ins['type']}**: {ins['finding']}")

        with sub_col2:
            st.subheader("🎯 Machine Learning Advisor")
            target_col = st.selectbox("Select Target Variable (Y):", options=df_active.columns.tolist())
            if st.button("Recommend ML Strategy"):
                rec = data_copilot.recommend_ml_models(df_active, target_col)
                if "error" in rec:
                    st.error(rec["error"])
                else:
                    st.success(f"**Problem Type:** `{rec['problem_type']}` (Cardinality: {rec['target_unique_values']})")
                    st.markdown(f"**Validation Protocol:** {rec['validation_strategy']}")
                    st.markdown(f"**Preprocessing:** {rec['preprocessing_pipeline']}")
                    st.markdown(f"**Metrics:** {', '.join(rec['primary_evaluation_metrics'])}")
                    st.markdown("**Top Recommended Architectures:**")
                    for m in rec["recommended_algorithms"]:
                        st.markdown(f"- **{m['name']}**: {m['rationale']}")
                    st.markdown(rec["ai_architect_advice"])

        st.markdown("---")
        st.subheader("💬 Natural Language Dataset Query")
        sample_queries = [
            "Show summary statistics",
            "Show top 5 rows",
            "Show correlation matrix",
            "Check for missing values"
        ]
        q_text = st.text_input("Ask a question about the dataset:", placeholder="e.g. 'Show summary statistics' or 'Show top 5 rows'")
        if st.button("Execute Natural Language Query") and q_text:
            res = data_copilot.query_data_natural_language(df_active, q_text)
            st.info(f"**Interpretation:** {res['interpretation']}")
            st.dataframe(res["data"], use_container_width=True)
            st.caption(res["text_response"])

# ==============================================================================
# TAB 3: RESEARCH COPILOT
# ==============================================================================
with tab3:
    st.header("📑 AI Research & Document Synthesizer")
    st.caption("Summarize technical documents, extract key computational concepts, and generate structured study guides.")

    research_copilot = ResearchCopilot(st.session_state.engine_manager)

    sample_research_text = (
        "Supervised machine learning algorithms map input feature vectors to target outcomes via parameterized "
        "function approximation. In high-dimensional spaces, models frequently experience the curse of dimensionality "
        "and overfit empirical training data. Regularization frameworks, including L1 Lasso (inducing sparsity) "
        "and L2 Ridge (weight shrinkage), mitigate variance and ensure out-of-sample generalization. "
        "Rigorous model evaluation requires leakage-free preprocessing pipelines and K-fold stratified cross-validation."
    )

    doc_input = st.text_area("Input Technical Document or Notes:", value=sample_research_text, height=150)
    summary_mode = st.radio("Summary Mode:", ["executive", "technical", "action_plan"], horizontal=True)

    r_col1, r_col2, r_col3 = st.columns(3)
    btn_sum = r_col1.button("📄 Generate Summary", use_container_width=True)
    btn_concepts = r_col2.button("🏷️ Extract Concepts", use_container_width=True)
    btn_guide = r_col3.button("🎓 Generate Study Guide", use_container_width=True)

    if btn_sum and doc_input:
        st.subheader(f"📑 {summary_mode.capitalize()} Summary")
        summary_res = research_copilot.summarize_text(doc_input, mode=summary_mode)
        st.markdown(summary_res["summary_content"])

    if btn_concepts and doc_input:
        st.subheader("🏷️ Identified Technical Concepts & Entities")
        concepts = research_copilot.extract_key_concepts(doc_input)
        for c in concepts:
            st.markdown(f"- **{c['concept']}** (`{c['category']}`): {c['context']}")

    if btn_guide:
        st.subheader("🎓 Comprehensive Study Guide")
        guide_res = research_copilot.generate_study_guide("Machine Learning Regularization & Generalization")
        st.markdown(guide_res["study_guide_markdown"])

# ==============================================================================
# TAB 4: PROMPT LAB
# ==============================================================================
with tab4:
    st.header("🔬 Prompt Engineering Laboratory")
    st.caption("Benchmark the 3 foundational prompting paradigms: Zero-Shot, Few-Shot, and Chain-of-Thought (CoT).")

    prompt_lab = PromptLab(st.session_state.engine_manager)
    task_keys = prompt_lab.get_prompt_templates()

    task_selection = st.selectbox(
        "Select Prompt Engineering Scenario:",
        options=list(task_keys.keys()),
        format_func=lambda k: f"{task_keys[k]['name']} ({task_keys[k]['category']})"
    )

    sample_prompt_input = st.text_input(
        "Target Input for Evaluation:",
        value="def sum_odds(numbers):\n    total = 0\n    for n in numbers:\n        if n % 2 != 0: total += n\n    return total"
    )

    if st.button("🚀 Compare Prompting Paradigms", type="primary"):
        lab_results = prompt_lab.compare_paradigms(task_selection, sample_prompt_input)

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("### 1️⃣ Zero-Shot Prompting")
            st.caption("Direct instruction without demonstration.")
            z = lab_results["zero_shot"]
            st.metric("Reasoning Depth", z["reasoning_depth"])
            st.metric("Estimated Tokens", z["token_estimate"])
            st.info(f"**Best For:** {z['best_for']}")
            with st.expander("View Prompt Template"):
                st.code(z["prompt"])
            st.markdown("**Output Response:**")
            st.markdown(z["response"])

        with p2:
            st.markdown("### 2️⃣ Few-Shot Prompting")
            st.caption("Exemplar-guided contextual demonstration.")
            f = lab_results["few_shot"]
            st.metric("Reasoning Depth", f["reasoning_depth"])
            st.metric("Estimated Tokens", f["token_estimate"])
            st.info(f"**Best For:** {f['best_for']}")
            with st.expander("View Prompt Template"):
                st.code(f["prompt"])
            st.markdown("**Output Response:**")
            st.markdown(f["response"])

        with p3:
            st.markdown("### 3️⃣ Chain-of-Thought (CoT)")
            st.caption("Multi-step reasoning decomposition.")
            c = lab_results["chain_of_thought"]
            st.metric("Reasoning Depth", c["reasoning_depth"])
            st.metric("Estimated Tokens", c["token_estimate"])
            st.info(f"**Best For:** {c['best_for']}")
            with st.expander("View Prompt Template"):
                st.code(c["prompt"])
            st.markdown("**Output Response:**")
            st.markdown(c["response"])

# ==============================================================================
# TAB 5: BENCHMARKS
# ==============================================================================
with tab5:
    st.header("⚡ AI Developer Tools Benchmark Matrix")
    st.caption("Comprehensive evaluation of OpenAI ChatGPT (GPT-4o), Google Gemini, Microsoft Copilot, and Claude 3.5 Sonnet.")

    bench_df = AIToolsBenchmark.get_benchmark_df()
    st.dataframe(bench_df, use_container_width=True)

    # Display Benchmark Chart
    chart_path = "charts/ai_tools_comparison.png"
    if not os.path.exists(chart_path):
        os.makedirs("charts", exist_ok=True)
        chart_path = AIToolsBenchmark.plot_benchmark_chart(chart_path)

    if os.path.exists(chart_path):
        st.image(chart_path, caption="Comparative Benchmark Performance: Coding, Debugging & Mathematical Reasoning", use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 In-Depth Tool Architectural Profiles")

    profiles = AIToolsBenchmark.get_tool_profiles()
    for tool_name, p in profiles.items():
        with st.expander(f"🔹 {tool_name} — {p['developer']}", expanded=False):
            st.markdown(f"**Flagship Model:** `{p['flagship_model']}`")
            st.markdown(f"**Context Window:** `{p['context_window']}`")
            st.markdown(f"**⚡ Architectural Superpower:** {p['superpower']}")
            st.markdown(f"**🎯 Optimal Engineering Use Cases:** {p['best_for']}")
            st.markdown(f"**💰 Pricing & Availability:** {p['pricing']}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    DevPulse AI Studio • Built for Codomax Internship Program (Module 5) • 
    Created by <b>srujan1-creator</b> • MIT Licensed
</div>
""", unsafe_allow_html=True)
