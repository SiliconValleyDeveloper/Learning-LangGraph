"""
Phase 10 · Lesson 1 — Person Finder (research → extract → reflect)

Inspired by LangChain's people-researcher agent:
  generate_queries → research_person → extract_profile → reflection
  (loop back to research when gaps remain)

What you will learn
-------------------
1. Multi-step research graphs beyond a single ReAct tool loop
2. Generate targeted search queries from a person + schema
3. Extract a structured profile from consolidated notes
4. Reflect on completeness and optionally run follow-up searches

Needs: Ollama running and an internet connection
Optional: TAVILY_API_KEY for Tavily search (falls back to DuckDuckGo)

Run:
    python Learning/10_person_finder/01_person_finder.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph

MAX_SEARCH_QUERIES = 3
MAX_SEARCH_RESULTS = 4
MAX_REFLECTION_STEPS = 1

DEFAULT_SCHEMA: dict[str, Any] = {
    "title": "Person",
    "description": "Public professional profile for a person",
    "type": "object",
    "required": ["full_name", "current_role", "current_company", "summary"],
    "properties": {
        "full_name": {"type": "string", "description": "Full name of the person"},
        "current_role": {"type": "string", "description": "Current job title or role"},
        "current_company": {
            "type": "string",
            "description": "Current employer or organization",
        },
        "location": {"type": "string", "description": "City / country if known"},
        "prior_companies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Previous employers",
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Notable skills or expertise areas",
        },
        "summary": {
            "type": "string",
            "description": "2-4 sentence public biography grounded in sources",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Source URLs used for this profile",
        },
    },
}


class PersonState(TypedDict):
    person: dict[str, str]
    user_notes: str
    extraction_schema: dict[str, Any]
    search_queries: list[str]
    research_notes: Annotated[list[str], lambda left, right: (left or []) + (right or [])]
    sources: Annotated[list[str], lambda left, right: list(dict.fromkeys((left or []) + (right or [])))]
    profile: NotRequired[dict[str, Any]]
    is_satisfactory: NotRequired[bool]
    reflection_steps_taken: int
    reply: NotRequired[str]


def _person_label(person: dict[str, str]) -> str:
    parts = []
    for key in ("name", "email", "company", "role", "linkedin"):
        value = (person.get(key) or "").strip()
        if value:
            parts.append(f"{key}: {value}")
    return " | ".join(parts) if parts else "unknown person"


def parse_person_message(message: str) -> tuple[dict[str, str], str]:
    """Parse free text or labeled fields into person dict + notes."""
    text = message.strip()
    person: dict[str, str] = {}
    notes = ""

    # Try JSON payload first
    if text.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                for key in ("name", "email", "company", "role", "linkedin"):
                    if payload.get(key):
                        person[key] = str(payload[key]).strip()
                notes = str(payload.get("notes") or payload.get("user_notes") or "").strip()
                if person:
                    return person, notes
        except json.JSONDecodeError:
            pass

    patterns = {
        "name": r"(?im)^\s*(?:name|full\s*name)\s*[:=]\s*(.+)$",
        "email": r"(?im)^\s*(?:email|e-mail)\s*[:=]\s*(.+)$",
        "company": r"(?im)^\s*(?:company|org|organization)\s*[:=]\s*(.+)$",
        "role": r"(?im)^\s*(?:role|title|job)\s*[:=]\s*(.+)$",
        "linkedin": r"(?im)^\s*(?:linkedin|linkedin\s*url)\s*[:=]\s*(.+)$",
        "notes": r"(?im)^\s*(?:notes|user\s*notes|context)\s*[:=]\s*(.+)$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if key == "notes":
                notes = value
            else:
                person[key] = value

    if not person.get("name"):
        # Treat first non-empty line / whole prompt as the name when unlabeled
        first = next((line.strip() for line in text.splitlines() if line.strip()), text)
        if ":" not in first or not person:
            person["name"] = first[:200]

    return person, notes


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def _llm_json(system: str, user: str) -> Any:
    llm = get_llm(temperature=0)
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    return _extract_json(content)


def _search_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    rows = list(DDGS().text(query, max_results=max_results))
    results: list[dict[str, str]] = []
    for row in rows:
        results.append(
            {
                "title": row.get("title") or "Untitled",
                "url": row.get("href") or row.get("url") or "",
                "snippet": row.get("body") or row.get("snippet") or "",
            }
        )
    return results


def _search_tavily(query: str, max_results: int) -> list[dict[str, str]]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    payload = client.search(query, max_results=max_results, topic="general")
    results: list[dict[str, str]] = []
    for row in payload.get("results") or []:
        results.append(
            {
                "title": row.get("title") or "Untitled",
                "url": row.get("url") or "",
                "snippet": row.get("content") or row.get("raw_content") or "",
            }
        )
    return results


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict[str, str]]:
    cleaned = query.strip()
    if not cleaned:
        return []
    if os.getenv("TAVILY_API_KEY"):
        try:
            return _search_tavily(cleaned, max_results)
        except Exception:  # noqa: BLE001 — fall back to DuckDuckGo
            pass
    return _search_ddg(cleaned, max_results)


def _seed_queries(person: dict[str, str]) -> list[str]:
    name = (person.get("name") or person.get("email") or "person").strip()
    company = (person.get("company") or "").strip()
    role = (person.get("role") or "").strip()
    seeds = [
        " ".join(part for part in (name, company, role) if part),
        f'"{name}" {company}'.strip(),
        f"{name} LinkedIn {company}".strip(),
    ]
    return [q for q in dict.fromkeys(seeds) if q][:MAX_SEARCH_QUERIES]


def _name_tokens(name: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", name) if len(token) > 1]


def _result_mentions_person(hit: dict[str, str], person: dict[str, str]) -> bool:
    name = (person.get("name") or "").strip()
    if not name:
        return True
    tokens = _name_tokens(name)
    if not tokens:
        return True
    blob = f"{hit.get('title', '')} {hit.get('snippet', '')} {hit.get('url', '')}".lower()
    # Require every name token to appear, or the full name phrase.
    if name.lower() in blob:
        return True
    return all(token in blob for token in tokens)


def _names_aligned(requested: str, extracted: str) -> bool:
    req = _name_tokens(requested)
    got = _name_tokens(extracted)
    if not req or not got:
        return False
    # Accept if last token matches and at least one other token overlaps,
    # or if all requested tokens appear in the extracted name.
    if all(token in got for token in req):
        return True
    if req[-1] == got[-1] and any(token in got for token in req[:-1]):
        return True
    return False


def generate_queries(state: PersonState) -> dict[str, Any]:
    person = state["person"]
    schema = state.get("extraction_schema") or DEFAULT_SCHEMA
    notes = state.get("user_notes") or ""
    name = (person.get("name") or "").strip()
    system = (
        "You generate web search queries to research ONE specific person. "
        f"Every query MUST include the exact name {name!r}. "
        "Do not substitute a different famous person. "
        f"Return ONLY JSON: {{\"queries\": [\"...\"]}} with up to "
        f"{MAX_SEARCH_QUERIES} specific queries."
    )
    user = (
        f"Person: {_person_label(person)}\n"
        f"User notes: {notes or '(none)'}\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )
    queries = _seed_queries(person)
    try:
        raw = _llm_json(system, user)
        llm_queries = [str(q).strip() for q in (raw.get("queries") or []) if str(q).strip()]
        # Keep only queries that still mention the target person.
        for query in llm_queries:
            if name and name.lower() not in query.lower() and not all(
                token in query.lower() for token in _name_tokens(name)
            ):
                continue
            if query not in queries:
                queries.append(query)
    except Exception:  # noqa: BLE001
        pass

    return {"search_queries": queries[:MAX_SEARCH_QUERIES]}


def research_person(state: PersonState) -> dict[str, Any]:
    person = state["person"]
    schema = state.get("extraction_schema") or DEFAULT_SCHEMA
    notes = state.get("user_notes") or ""
    queries = state.get("search_queries") or []
    target = (person.get("name") or "").strip()

    collected: list[dict[str, str]] = []
    source_urls: list[str] = []
    for query in queries:
        try:
            hits = web_search(query)
        except Exception as exc:  # noqa: BLE001
            collected.append(
                {
                    "title": "Search error",
                    "url": "",
                    "snippet": f"Query {query!r} failed: {exc}",
                }
            )
            continue
        for hit in hits:
            if not _result_mentions_person(hit, person):
                continue
            collected.append(hit)
            if hit.get("url"):
                source_urls.append(hit["url"])

    if not collected:
        return {
            "research_notes": [
                "NO MATCHING PUBLIC RESULTS. "
                f"Could not find web snippets that clearly mention {target or _person_label(person)}. "
                f"Queries tried: {queries}. "
                "Do not invent a biography from model memory."
            ],
            "sources": [],
        }

    blocks = []
    for index, hit in enumerate(collected, start=1):
        blocks.append(
            f"{index}. {hit.get('title')}\nURL: {hit.get('url')}\nSnippet: {hit.get('snippet')}"
        )
    source_blob = "\n\n".join(blocks)

    system = (
        f"You take research notes ONLY about {target or 'the requested person'}. "
        "Use ONLY the provided search snippets. "
        "If snippets are about a different person, ignore them. "
        "If evidence is weak, say what is unknown. "
        "Never substitute a more famous person. Return plain text notes, not JSON."
    )
    user = (
        f"Target person: {_person_label(person)}\n"
        f"User notes: {notes or '(none)'}\n"
        f"Schema fields to cover:\n{json.dumps(schema, indent=2)}\n\n"
        f"Search results:\n{source_blob}"
    )
    llm = get_llm(temperature=0)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    note_text = response.content if isinstance(response.content, str) else str(response.content)
    return {"research_notes": [note_text], "sources": source_urls}


def extract_profile(state: PersonState) -> dict[str, Any]:
    schema = state.get("extraction_schema") or DEFAULT_SCHEMA
    notes = "\n\n---\n\n".join(state.get("research_notes") or [])
    sources = state.get("sources") or []
    person = state["person"]
    target = (person.get("name") or "").strip()
    system = (
        f"Extract a JSON object for ONLY this person: {target}. "
        "full_name MUST be this person (or null if unknown). "
        "Use ONLY the research notes. If notes say there is no match, or describe someone else, "
        "set current_role/current_company/summary to null/empty and explain the lack of evidence in summary. "
        "Never invent facts from model memory. Never return a different famous person. "
        "Return ONLY JSON matching the schema."
    )
    user = (
        f"Target person: {_person_label(person)}\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Known source URLs:\n{json.dumps(sources, indent=2)}\n\n"
        f"Notes:\n{notes}"
    )
    try:
        profile = _llm_json(system, user)
        if not isinstance(profile, dict):
            profile = {"summary": str(profile), "sources": sources}
    except Exception as exc:  # noqa: BLE001
        profile = {
            "full_name": target,
            "summary": f"Could not extract a structured profile: {exc}",
            "sources": sources,
            "raw_notes": notes[:2000],
        }

    if not isinstance(profile, dict):
        profile = {"full_name": target, "summary": str(profile), "sources": sources}

    extracted_name = str(profile.get("full_name") or "")
    no_match_notes = "NO MATCHING PUBLIC RESULTS" in notes.upper()
    wrong_person = bool(target and extracted_name and not _names_aligned(target, extracted_name))

    if no_match_notes or wrong_person or not sources:
        profile = {
            "full_name": target or extracted_name or None,
            "current_role": person.get("role") or None,
            "current_company": person.get("company") or None,
            "location": None,
            "prior_companies": [],
            "skills": [],
            "summary": (
                f"No reliable public web evidence was found for {target or 'this person'}. "
                "The model did not invent a biography. Try adding LinkedIn URL, email, "
                "or more distinctive company/role details."
                if no_match_notes or not sources
                else (
                    f"Search results did not support profile '{extracted_name}' for requested "
                    f"person '{target}'. Refusing to return a mismatched biography."
                )
            ),
            "sources": sources,
            "match_confidence": "low",
        }
    else:
        profile["full_name"] = target or extracted_name
        profile["match_confidence"] = "medium"
        if sources and not profile.get("sources"):
            profile["sources"] = sources

    return {"profile": profile}


def reflection(state: PersonState) -> dict[str, Any]:
    schema = state.get("extraction_schema") or DEFAULT_SCHEMA
    profile = state.get("profile") or {}
    person = state["person"]
    target = (person.get("name") or "").strip()
    system = (
        "Judge whether the extracted profile is about the REQUESTED person and adequately "
        "fills required schema fields using only public web evidence. "
        "If full_name is a different person, is_satisfactory must be false. "
        "Return ONLY JSON with keys: "
        "is_satisfactory (bool), missing_fields (string[]), search_queries (string[]), "
        "reasoning (string). Follow-up queries must include the requested name."
    )
    user = (
        f"Requested person: {_person_label(person)}\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Extracted profile:\n{json.dumps(profile, indent=2)}"
    )
    low_confidence = str(profile.get("match_confidence") or "") == "low"
    try:
        raw = _llm_json(system, user)
        is_satisfactory = bool(raw.get("is_satisfactory")) and not low_confidence
        followups = [str(q).strip() for q in (raw.get("search_queries") or []) if str(q).strip()]
        reasoning = str(raw.get("reasoning") or "")
        missing = [str(f) for f in (raw.get("missing_fields") or [])]
    except Exception as exc:  # noqa: BLE001
        is_satisfactory = not low_confidence
        followups = []
        reasoning = f"Reflection parse failed; accepting current profile ({exc})"
        missing = []

    if target and profile.get("full_name") and not _names_aligned(
        target, str(profile.get("full_name"))
    ):
        is_satisfactory = False
        missing = list(dict.fromkeys([*missing, "full_name"]))
        reasoning = (
            f"Extracted name {profile.get('full_name')!r} does not match requested "
            f"{target!r}. {reasoning}"
        ).strip()

    if low_confidence:
        is_satisfactory = True  # stop looping; honest "not found" is complete
        reasoning = reasoning or "No reliable public match; stopping without invention."

    steps = int(state.get("reflection_steps_taken") or 0)
    update: dict[str, Any] = {
        "is_satisfactory": is_satisfactory,
        "reflection_steps_taken": steps if is_satisfactory else steps + 1,
    }
    if not is_satisfactory:
        seeded = [
            q
            for q in followups
            if not target
            or target.lower() in q.lower()
            or all(token in q.lower() for token in _name_tokens(target))
        ] or _seed_queries(person)
        update["search_queries"] = seeded[:MAX_SEARCH_QUERIES]

    profile = dict(profile)
    profile["_reflection"] = {
        "is_satisfactory": is_satisfactory,
        "missing_fields": missing,
        "reasoning": reasoning,
        "steps_taken": update["reflection_steps_taken"],
    }
    update["profile"] = profile

    reply = (
        f"# Person profile\n\n```json\n{json.dumps({k: v for k, v in profile.items() if k != '_reflection'}, indent=2)}\n```\n"
    )
    if reasoning:
        reply += f"\n**Reflection:** {reasoning}\n"
    sources = profile.get("sources") or state.get("sources") or []
    if sources:
        reply += "\n**Sources:**\n" + "\n".join(f"- {url}" for url in sources[:12])
    update["reply"] = reply
    return update


def route_from_reflection(state: PersonState) -> str:
    if state.get("is_satisfactory"):
        return END
    if int(state.get("reflection_steps_taken") or 0) <= MAX_REFLECTION_STEPS:
        return "research_person"
    return END


def build_graph(checkpointer=None):
    builder = StateGraph(PersonState)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("research_person", research_person)
    builder.add_node("extract_profile", extract_profile)
    builder.add_node("reflection", reflection)
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "research_person")
    builder.add_edge("research_person", "extract_profile")
    builder.add_edge("extract_profile", "reflection")
    builder.add_conditional_edges(
        "reflection",
        route_from_reflection,
        {END: END, "research_person": "research_person"},
    )
    return builder.compile(checkpointer=checkpointer)


def initial_state_from_message(message: str) -> PersonState:
    person, notes = parse_person_message(message)
    return {
        "person": person,
        "user_notes": notes,
        "extraction_schema": DEFAULT_SCHEMA,
        "search_queries": [],
        "research_notes": [],
        "sources": [],
        "reflection_steps_taken": 0,
    }


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="Person Finder")
    demo = initial_state_from_message(
        "Name: Harrison Chase\nCompany: LangChain\nRole: CEO\nNotes: Focus on public professional info"
    )
    result = graph.invoke(demo)
    print(f"\n{result.get('reply') or json.dumps(result.get('profile'), indent=2)}\n")
