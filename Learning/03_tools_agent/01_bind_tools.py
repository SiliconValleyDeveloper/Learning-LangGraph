"""
Phase 3 · Lesson 1 — Bind tools to the LLM (one shot)

What you will learn
-------------------
1. Define tools with @tool (name, description, typed args)
2. llm.bind_tools([...]) — model may request a tool instead of plain text
3. AIMessage.tool_calls — structured {name, args, id}
4. Why one shot is NOT enough — you still need to run the tool and loop

Needs: Ollama running

Run:
    python 03_tools_agent/01_bind_tools.py
"""

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


def main() -> None:
    require_ollama()
    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools([multiply, add])

    messages = [
        SystemMessage(
            content=(
                "You are a calculator assistant. "
                "When the user asks for arithmetic, call a tool. "
                "Do not invent numbers."
            )
        ),
        HumanMessage(content="What is 12 multiplied by 7?"),
    ]

    print("\n=== One LLM call with tools bound ===\n")
    ai = llm_with_tools.invoke(messages)
    print(f"  content   : {ai.content!r}")
    print(f"  tool_calls: {ai.tool_calls}")

    if not ai.tool_calls:
        print(
            "\n  Model answered in text only (no tool_calls).\n"
            "  Try again, or move to 02_react_agent.py which loops properly.\n"
        )
        return

    # Manually run the first tool (Lesson 2 automates this with ToolNode)
    call = ai.tool_calls[0]
    selected = {"multiply": multiply, "add": add}[call["name"]]
    result = selected.invoke(call["args"])
    print(f"\n  Ran tool '{call['name']}'({call['args']}) → {result}")
    print(
        "\n  Notice: we still need to send this result BACK to the LLM\n"
        "  so it can write a final answer. That loop = ReAct (next lesson).\n"
    )


if __name__ == "__main__":
    main()
