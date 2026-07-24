"""Reliable live internet search (Tavily preferred, DuckDuckGo fallback)."""

from __future__ import annotations

import os
import re
import time
from typing import Any
from urllib.parse import urlparse

# Low-signal / spammy hosts to drop.
_BLOCKED_HOST_FRAGMENTS = (
    "pinterest.",
    "quora.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com/r/",
)

_TIME_SENSITIVE = re.compile(
    r"\b(today|latest|current|now|recent|who won|score|price|news|202[4-9]|2026)\b",
    re.I,
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _clean_hit(item: dict[str, Any]) -> dict[str, str] | None:
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or item.get("href") or item.get("link") or "").strip()
    snippet = str(
        item.get("snippet") or item.get("body") or item.get("content") or ""
    ).strip()
    if not url.startswith("http"):
        return None
    if len(snippet) < 40 and len(title) < 12:
        return None
    host = _host(url)
    if any(bad in host or bad in url.lower() for bad in _BLOCKED_HOST_FRAGMENTS):
        return None
    if title.lower() in {"untitled", "search error"}:
        return None
    return {
        "title": title[:180] or host or "Source",
        "url": url,
        "snippet": snippet[:480],
        "host": host,
    }


def _dedupe(hits: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for hit in hits:
        key = hit.get("url") or hit.get("title", "").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _tavily_search(query: str, max_results: int) -> list[dict[str, str]]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return []
    from tavily import TavilyClient

    client = TavilyClient(api_key=key)
    payload = client.search(
        query,
        max_results=max_results,
        include_answer=False,
        search_depth="advanced",
    )
    hits: list[dict[str, str]] = []
    for item in payload.get("results") or []:
        cleaned = _clean_hit(item)
        if cleaned:
            hits.append(cleaned)
    return hits


def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    raw = list(DDGS().text(query, max_results=max_results + 3))
    hits: list[dict[str, str]] = []
    for item in raw:
        cleaned = _clean_hit(item)
        if cleaned:
            hits.append(cleaned)
    return hits


def _search_once(query: str, max_results: int) -> list[dict[str, str]]:
    errors: list[str] = []
    try:
        hits = _tavily_search(query, max_results)
        if hits:
            return hits
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tavily:{exc}")
    try:
        hits = _ddg_search(query, max_results)
        if hits:
            return hits
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ddg:{exc}")
    if errors:
        return [
            {
                "title": "Search temporarily unavailable",
                "url": "",
                "snippet": "Providers failed: " + " | ".join(errors)[:300],
                "host": "",
            }
        ]
    return []


def build_search_queries(question: str, rewritten: str | None = None) -> list[str]:
    """Build 1–3 complementary queries for better recall."""
    base = (rewritten or question).strip()
    q = question.strip()
    queries = [base]
    if q and q.lower() != base.lower():
        queries.append(q)
    if _TIME_SENSITIVE.search(q) and "2026" not in base:
        queries.append(f"{base} 2026")
    # Keep unique, short list
    unique: list[str] = []
    for item in queries:
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned and cleaned.lower() not in {u.lower() for u in unique}:
            unique.append(cleaned[:160])
    return unique[:3]


def search_web(
    query: str,
    *,
    max_results: int = 5,
    extra_queries: list[str] | None = None,
    retries: int = 2,
) -> list[dict[str, str]]:
    """Search with multi-query + retries + filtering for higher reliability."""
    queries = [query.strip()] if query.strip() else []
    for item in extra_queries or []:
        if item.strip() and item.strip().lower() not in {q.lower() for q in queries}:
            queries.append(item.strip())
    if not queries:
        return []

    pooled: list[dict[str, str]] = []
    for q in queries:
        attempt_hits: list[dict[str, str]] = []
        for attempt in range(max(1, retries)):
            attempt_hits = _search_once(q, max_results=max_results)
            # Real hits have URLs
            if any(h.get("url") for h in attempt_hits):
                break
            time.sleep(0.35 * (attempt + 1))
        pooled.extend(h for h in attempt_hits if h.get("url"))

    ranked = _dedupe(pooled)
    # Prefer longer snippets / official-looking hosts
    ranked.sort(
        key=lambda h: (
            0 if any(x in h.get("host", "") for x in ("wikipedia.org", "github.com", "aws.amazon.com", "langchain.com", "openai.com")) else 1,
            -len(h.get("snippet") or ""),
        )
    )
    return ranked[:max_results]


def format_web_context(hits: list[dict[str, str]]) -> str:
    if not hits:
        return ""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[W{index}] {hit['title']}\nURL: {hit['url']}\n{hit['snippet']}"
        )
    return "\n\n".join(blocks)


def web_result_summary(hits: list[dict[str, Any]]) -> str:
    usable = [h for h in hits if h.get("url")]
    if not usable:
        return "No reliable web results"
    titles = [str(h.get("title") or "")[:50] for h in usable[:3]]
    return f"{len(usable)} sources · " + " · ".join(t for t in titles if t)


def evidence_overlap(question: str, text: str) -> float:
    """Simple token overlap score in [0, 1] for relevance checks."""
    q_tokens = {t for t in re.findall(r"[a-z0-9]{3,}", question.lower()) if t not in {
        "the", "and", "for", "what", "how", "who", "when", "where", "with", "from", "that", "this"
    }}
    if not q_tokens:
        return 0.0
    blob = text.lower()
    hits = sum(1 for t in q_tokens if t in blob)
    return hits / max(len(q_tokens), 1)
