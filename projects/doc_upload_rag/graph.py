"""Simple retrieve → generate graph over a user's uploaded documents."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from Learning.llm import get_llm
from projects.doc_upload_rag.store import get_workspace


class AskState(TypedDict):
    workspace_id: str
    question: str
    context: str
    sources: list[str]
    answer: str
    chunk_previews: list[dict[str, Any]]


def _retrieve(state: AskState) -> dict[str, Any]:
    store = get_workspace(state["workspace_id"])
    if store.document_count == 0:
        return {
            "context": "",
            "sources": [],
            "chunk_previews": [],
        }

    scored = store.retrieve(state["question"], k=4)
    blocks: list[str] = []
    sources: list[str] = []
    previews: list[dict[str, Any]] = []
    for index, (document, score) in enumerate(scored, start=1):
        source = str(document.metadata.get("source", "unknown"))
        sources.append(source)
        blocks.append(f"[{index}] Source: {source}\n{document.page_content}")
        previews.append(
            {
                "source": source,
                "score": round(score, 4),
                "preview": document.page_content[:220],
            }
        )
    return {
        "context": "\n\n".join(blocks),
        "sources": sorted(set(sources)),
        "chunk_previews": previews,
    }


def _generate(state: AskState) -> dict[str, Any]:
    if not state.get("context"):
        return {
            "answer": (
                "I don't have any uploaded documents in this workspace yet. "
                "Upload a .md or .txt file, then ask again."
            )
        }

    llm = get_llm(temperature=0.1, reasoning=False)
    messages = [
        SystemMessage(
            content=(
                "Answer using ONLY the provided document excerpts. "
                "Cite sources like [filename.md]. "
                "If the excerpts do not contain the answer, say you cannot find it "
                "in the uploaded documents."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"Excerpts:\n{state['context']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"answer": str(response.content).strip()}


def build_ask_graph():
    graph = StateGraph(AskState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_ASK_GRAPH = None


def ask(workspace_id: str, question: str) -> dict[str, Any]:
    global _ASK_GRAPH
    if _ASK_GRAPH is None:
        _ASK_GRAPH = build_ask_graph()
    result = _ASK_GRAPH.invoke(
        {
            "workspace_id": workspace_id,
            "question": question,
            "context": "",
            "sources": [],
            "answer": "",
            "chunk_previews": [],
        }
    )
    store = get_workspace(workspace_id)
    return {
        "answer": result["answer"],
        "sources": result.get("sources") or [],
        "chunk_previews": result.get("chunk_previews") or [],
        "workspace": {
            "id": workspace_id,
            "document_count": store.document_count,
            "chunk_count": store.chunk_count,
        },
    }
