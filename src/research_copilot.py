"""Research Copilot Module — DevPulse AI Studio
Provides AI-assisted research synthesis, academic summarization, concept extraction,
and automated study guide / productivity generation.
"""

import re
from typing import Dict, Any, List, Optional
from src.ai_engines import AIEngineManager


class ResearchCopilot:
    """Intelligent Assistant for technical documentation, academic papers, and research."""

    def __init__(self, engine_manager: Optional[AIEngineManager] = None):
        self.engine_manager = engine_manager or AIEngineManager()

    def summarize_text(self, text: str, mode: str = "executive") -> Dict[str, Any]:
        """Summarize research text or documentation into structured executive or technical format."""
        word_count = len(text.strip().split())
        char_count = len(text.strip())

        system_prompts = {
            "executive": "You are a Principal AI Research Scientist. Summarize the text into an Executive Abstract, 3 Key Findings, and Strategic Impact.",
            "technical": "You are a Senior Machine Learning Researcher. Break down methodology, mathematical foundations, architectural components, and computational complexity.",
            "action_plan": "You are an Agile Engineering Lead. Convert the findings into a numbered, actionable step-by-step implementation checklist."
        }
        chosen_sys = system_prompts.get(mode, system_prompts["executive"])

        prompt = (
            f"Provide a structured {mode.upper()} summary for the following text:\n\n"
            f"{text}\n\n"
            f"Original word count: {word_count} words."
        )

        ai_summary = self.engine_manager.generate(prompt, chosen_sys)

        # Fallback heuristic if offline engine returned general output
        if not self.engine_manager.get_active_engine().is_live:
            # Generate deterministic structured summary
            sentences = [s.strip() for s in re.split(r'[.\n]+', text) if len(s.strip()) > 20]
            top_findings = sentences[:3] if len(sentences) >= 3 else sentences

            findings_markdown = "\n".join([f"{i+1}. **Point {i+1}:** {f}." for i, f in enumerate(top_findings)])
            ai_summary = (
                f"### 📑 DevPulse AI Research Synthesis ({mode.capitalize()} Mode)\n\n"
                f"**Executive Abstract:**\n"
                f"This document articulates critical insights across {word_count} words. The primary contribution centers "
                f"on robust computational workflows, data-driven optimization, and systematic engineering pipelines.\n\n"
                f"**Core Findings:**\n{findings_markdown}\n\n"
                f"**Productivity & Implementation Impact:**\n"
                f"- High feasibility for immediate deployment in Python environments.\n"
                f"- Reproducible methodology ensuring rigorous benchmark validation."
            )

        return {
            "mode": mode,
            "original_word_count": word_count,
            "original_char_count": char_count,
            "summary_content": ai_summary
        }

    def extract_key_concepts(self, text: str) -> List[Dict[str, str]]:
        """Extract technical terms, algorithms, metrics, and definitions from text."""
        # Regex extraction of capitalized multi-word terms or acronyms
        acronyms = set(re.findall(r"\b[A-Z]{2,6}\b", text))
        tech_patterns = [
            "machine learning", "deep learning", "neural network", "transformer", "gradient descent",
            "cross-validation", "overfitting", "underfitting", "regularization", "hyperparameter",
            "supervised learning", "unsupervised learning", "classification", "regression",
            "scikit-learn", "pandas", "numpy", "stream-lit", "api", "prompt engineering"
        ]

        detected_concepts = []
        text_lower = text.lower()

        for term in tech_patterns:
            if term in text_lower:
                detected_concepts.append({
                    "concept": term.title(),
                    "category": "Core Methodology / Tooling",
                    "context": f"Identified in context: '{term}' is central to modern AI workflows."
                })

        for ac in acronyms:
            detected_concepts.append({
                "concept": ac,
                "category": "Technical Acronym",
                "context": f"Standard domain abbreviation appearing in source text."
            })

        # Deduplicate
        unique_concepts = []
        seen = set()
        for c in detected_concepts:
            if c["concept"] not in seen:
                seen.add(c["concept"])
                unique_concepts.append(c)

        return unique_concepts[:10]

    def generate_study_guide(self, topic: str) -> Dict[str, Any]:
        """Generate structured FAQs and study flashcards for a specified topic or document."""
        prompt = (
            f"Generate an educational study guide for the topic: '{topic}'.\n"
            f"Include: 1) Core Definition, 2) 3 High-Yield Q&As, 3) Best Practices & Common Pitfalls."
        )
        guide_text = self.engine_manager.generate(prompt, "You are a University Professor in Computer Science and Artificial Intelligence.")

        if not self.engine_manager.get_active_engine().is_live:
            guide_text = (
                f"### 🎓 DevPulse Comprehensive Study Guide: *{topic}*\n\n"
                f"#### 1. Core Principle\n"
                f"{topic} encompasses foundational methodologies in contemporary computational intelligence, "
                f"enabling developers and data scientists to automate complex inference and optimize systems.\n\n"
                f"#### 2. High-Yield Question & Answer Cards\n"
                f"- **Q: Why is {topic} critical in production environments?**\n"
                f"  *A: It standardizes automated reasoning, reduces cycle times, and improves reproducibility.*\n"
                f"- **Q: What is the most common pitfall when implementing {topic}?**\n"
                f"  *A: Premature optimization without establishing a baseline metric or test suite.*\n"
                f"- **Q: How do we measure the efficacy of {topic}?**\n"
                f"  *A: Quantitative benchmarks (accuracy, latency, F1-score) alongside qualitative peer review.*\n\n"
                f"#### 3. Best Practices Checklist\n"
                f"✅ Implement modular, single-responsibility functions.\n"
                f"✅ Prevent data leakage during preprocessing.\n"
                f"✅ Benchmark baseline before introducing complex models."
            )

        return {
            "topic": topic,
            "study_guide_markdown": guide_text
        }
