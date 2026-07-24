"""Shared local LLM helper (Ollama — no OpenAI / API key).

Prereqs:
  1. Ollama installed: https://ollama.com
  2. A model pulled, e.g.  ollama pull qwen3:8b
  3. Ollama running (usually automatic on macOS)

Config via .env (optional):
  OLLAMA_MODEL=qwen3:8b
  OLLAMA_BASE_URL=http://localhost:11434
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

DEFAULT_MODEL = "qwen3:8b"


def get_llm(*, temperature: float = 0.3, reasoning: bool = False) -> ChatOllama:
    """Return a ChatOllama client configured from .env.

    reasoning=False turns off chain-of-thought on models like qwen3
    (faster + cleaner labels for routing demos).
    """
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        reasoning=reasoning,
    )


def require_ollama() -> None:
    """Exit with a clear message if Ollama is not reachable."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            if resp.status != 200:
                raise OSError(f"unexpected status {resp.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(
            "Cannot reach Ollama.\n"
            "  1. Install: https://ollama.com\n"
            "  2. Start the Ollama app (or: ollama serve)\n"
            f"  3. Pull a model: ollama pull {model}\n"
            f"  Tried: {base_url}\n"
            f"  Error: {exc}"
        ) from exc
