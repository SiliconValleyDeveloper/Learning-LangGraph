"""
Phase 11 · Lesson 1 — LLM basics with local Ollama

What you will learn
-------------------
1. An LLM receives messages and predicts the next tokens
2. System messages set behavior; user messages carry the task
3. Temperature changes sampling, not knowledge
4. Model output is not automatically factual or grounded

Run:
    python Learning/11_rag_llm_ecosystem/01_llm_basics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama


def ask(question: str, *, temperature: float) -> str:
    """Send two chat messages to the local language model."""
    llm = get_llm(temperature=temperature)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a concise LLM tutor. Explain concepts in plain language. "
                    "Do not claim that generated text is guaranteed to be factual."
                )
            ),
            HumanMessage(content=question),
        ]
    )
    return response.content if isinstance(response.content, str) else str(response.content)


if __name__ == "__main__":
    require_ollama()

    prompt = "What is the difference between a token and a word?"
    print("=== Messages ===")
    print("system: You are a concise LLM tutor ...")
    print(f"user:   {prompt}\n")

    print("=== Deterministic-style run (temperature=0) ===")
    print(ask(prompt, temperature=0))

    print("\n=== More varied run (temperature=0.8) ===")
    print(ask("Give one analogy for how an LLM predicts text.", temperature=0.8))

    print(
        "\nKey idea: the LLM generated both answers from its parameters. "
        "It did not search or read our documents. RAG will add that grounding."
    )
