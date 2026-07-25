"""Filings ingest + pgvector retrieve + dynamic rerank (F6).

CLI:
    python -m projects.finance_agent.filings ingest
    python -m projects.finance_agent.filings list RELIANCE
    python -m projects.finance_agent.filings search RELIANCE "What did management say about debt?"
    python -m projects.finance_agent.filings status
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from projects.advanced_chatbot.rerank import rerank_chunks
from projects.finance_agent import db
from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.embeddings import embed_texts, vector_literal
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.filings")
DATA_DIR = Path(__file__).resolve().parent / "data"
FILINGS_DIR = DATA_DIR / "filings"

_SYMBOL_RE = re.compile(r"(?im)^Symbol:\s*([A-Z0-9.&-]+)")
_EXCHANGE_RE = re.compile(r"(?im)Exchange:\s*(NSE|BSE|OTHER)")
_DOC_TYPE_RE = re.compile(r"(?im)^Document type:\s*([a-z0-9_]+)")
_TITLE_RE = re.compile(r"(?m)^(.{8,120})$")


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        add_start_index=True,
    )


def _parse_header(text: str, filename: str) -> dict[str, str]:
    symbol_m = _SYMBOL_RE.search(text)
    exchange_m = _EXCHANGE_RE.search(text)
    doc_type_m = _DOC_TYPE_RE.search(text)
    title_m = _TITLE_RE.search(text)
    symbol = (symbol_m.group(1) if symbol_m else filename.split("_")[0]).upper()
    return {
        "symbol": symbol,
        "exchange": (exchange_m.group(1) if exchange_m else "NSE").upper(),
        "doc_type": (doc_type_m.group(1) if doc_type_m else "filing").lower(),
        "title": (title_m.group(1).strip() if title_m else filename)[:200],
    }


def sample_filing_paths() -> list[Path]:
    if not FILINGS_DIR.is_dir():
        return []
    return sorted(FILINGS_DIR.glob("*.txt"))


def upsert_document(
    *,
    symbol: str,
    exchange: str,
    title: str,
    filename: str,
    content_text: str,
    doc_type: str = "filing",
    source: str = "sample",
    metadata: dict[str, Any] | None = None,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    text = content_text.strip()
    if not text:
        raise ValueError(f"Empty filing: {filename}")

    chunks = _splitter().split_text(text)
    vectors, embed_backend = embed_texts(chunks, config=cfg)
    meta = dict(metadata or {})
    meta["embed_backend"] = embed_backend
    meta["chunk_count"] = len(chunks)

    with db.connect(cfg) as conn:
        existing = conn.execute(
            """
            SELECT id FROM finance_documents
            WHERE symbol = %s AND filename = %s AND source = %s
            LIMIT 1
            """,
            (symbol.upper(), filename, source),
        ).fetchone()
        doc_id = str(existing[0]) if existing else str(uuid4())

        if existing:
            conn.execute(
                """
                UPDATE finance_documents SET
                    exchange = %s,
                    doc_type = %s,
                    title = %s,
                    mime_type = 'text/plain',
                    content_text = %s,
                    metadata = %s::jsonb,
                    fetched_at = now(),
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    exchange.upper(),
                    doc_type,
                    title,
                    text,
                    json.dumps(meta),
                    doc_id,
                ),
            )
            conn.execute(
                "DELETE FROM finance_chunks WHERE document_id = %s",
                (doc_id,),
            )
        else:
            conn.execute(
                """
                INSERT INTO finance_documents (
                    id, symbol, exchange, doc_type, title, filename,
                    mime_type, content_text, metadata, source, fetched_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    'text/plain', %s, %s::jsonb, %s, now()
                )
                """,
                (
                    doc_id,
                    symbol.upper(),
                    exchange.upper(),
                    doc_type,
                    title,
                    filename,
                    text,
                    json.dumps(meta),
                    source,
                ),
            )

        from projects.finance_agent.embeddings import vectors_as_literals

        literals = vectors_as_literals(vectors)
        for index, (chunk, lit) in enumerate(zip(chunks, literals)):
            conn.execute(
                """
                INSERT INTO finance_chunks (
                    id, document_id, chunk_index, content, embedding,
                    metadata, source, fetched_at
                ) VALUES (
                    %s, %s, %s, %s, %s::vector, %s::jsonb, %s, now()
                )
                """,
                (
                    str(uuid4()),
                    doc_id,
                    index,
                    chunk,
                    lit,
                    json.dumps(
                        {
                            "symbol": symbol.upper(),
                            "filename": filename,
                            "title": title,
                            "doc_type": doc_type,
                            "chunk_index": index,
                        }
                    ),
                    source,
                ),
            )

    log.info(
        "filing_upserted",
        extra={
            "symbol": symbol.upper(),
            "file_name": filename,
            "chunks": len(chunks),
            "embed_backend": embed_backend,
        },
    )
    return {
        "id": doc_id,
        "symbol": symbol.upper(),
        "filename": filename,
        "chunk_count": len(chunks),
        "embed_backend": embed_backend,
    }


def ingest_sample(*, config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    paths = sample_filing_paths()
    if not paths:
        raise FileNotFoundError(f"No sample filings in {FILINGS_DIR}")
    results = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        header = _parse_header(text, path.name)
        results.append(
            upsert_document(
                symbol=header["symbol"],
                exchange=header["exchange"],
                title=header["title"],
                filename=path.name,
                content_text=text,
                doc_type=header["doc_type"],
                source="sample",
                metadata={"path": str(path.name)},
                config=cfg,
            )
        )
    return {
        "source": "sample",
        "documents": len(results),
        "chunks": sum(int(r["chunk_count"]) for r in results),
        "embed_backend": results[0]["embed_backend"] if results else None,
        "items": results,
    }


def list_documents(
    symbol: str | None = None,
    *,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    clauses = ["1=1"]
    params: list[Any] = []
    if symbol:
        clauses.append("symbol = %s")
        params.append(symbol.upper())
    where = " AND ".join(clauses)
    with db.connect(cfg) as conn:
        rows = conn.execute(
            f"""
            SELECT d.id, d.symbol, d.exchange, d.doc_type, d.title, d.filename,
                   d.source, d.fetched_at,
                   (SELECT count(*) FROM finance_chunks c WHERE c.document_id = d.id)
            FROM finance_documents d
            WHERE {where}
            ORDER BY d.symbol, d.fetched_at DESC
            """,
            params,
        ).fetchall()
    docs = [
        {
            "id": str(r[0]),
            "symbol": r[1],
            "exchange": r[2],
            "doc_type": r[3],
            "title": r[4],
            "filename": r[5],
            "source": r[6],
            "fetched_at": str(r[7]) if r[7] else None,
            "chunk_count": int(r[8] or 0),
        }
        for r in rows
    ]
    return {"count": len(docs), "documents": docs}


def retrieve_candidates(
    query: str,
    *,
    symbol: str | None = None,
    k: int | None = None,
    config: FinanceConfig | None = None,
) -> tuple[list[dict[str, Any]], str]:
    cfg = config or load_config()
    limit = k if k is not None else cfg.retrieve_candidates
    query_vec, embed_backend = vector_literal(query, config=cfg)
    clauses = ["c.embedding IS NOT NULL"]
    params: list[Any] = [query_vec]
    if symbol:
        clauses.append("d.symbol = %s")
        params.append(symbol.upper())
    where = " AND ".join(clauses)
    params.extend([query_vec, limit])
    with db.connect(cfg) as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.document_id, d.filename, d.title, d.symbol, d.doc_type,
                   c.content, 1 - (c.embedding <=> %s::vector) AS score, c.metadata
            FROM finance_chunks c
            JOIN finance_documents d ON d.id = c.document_id
            WHERE {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        ).fetchall()
    candidates = [
        {
            "chunk_id": str(r[0]),
            "document_id": str(r[1]),
            "source": r[2] or "filing",
            "title": r[3],
            "symbol": r[4],
            "doc_type": r[5],
            "content": r[6],
            "score": float(r[7] or 0.0),
            "metadata": r[8] or {},
        }
        for r in rows
    ]
    return candidates, embed_backend


def search_filings(
    query: str,
    *,
    symbol: str | None = None,
    top_k: int | None = None,
    candidates: int | None = None,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    """Retrieve → dynamic rerank (reuse advanced_chatbot.rerank)."""
    cfg = config or load_config()
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    hits, embed_backend = retrieve_candidates(
        q, symbol=symbol, k=candidates or cfg.retrieve_candidates, config=cfg
    )
    ranked, rerank_backend = rerank_chunks(q, hits, top_k=top_k or cfg.rerank_top_k)
    return {
        "query": q,
        "symbol": symbol.upper() if symbol else None,
        "embed_backend": embed_backend,
        "rerank_backend": rerank_backend,
        "retrieve_candidates": len(hits),
        "count": len(ranked),
        "chunks": [
            {
                "rank": item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "source": item.get("source"),
                "title": item.get("title"),
                "symbol": item.get("symbol"),
                "doc_type": item.get("doc_type"),
                "content": item.get("content"),
                "preview": str(item.get("content") or "")[:280],
                "vector_score": item.get("vector_score"),
                "rerank_score": item.get("rerank_score"),
            }
            for item in ranked
        ],
    }


def status(*, config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    with db.connect(cfg) as conn:
        docs = conn.execute("SELECT count(*) FROM finance_documents").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM finance_chunks").fetchone()[0]
        by_sym = conn.execute(
            """
            SELECT symbol, count(*)
            FROM finance_documents
            WHERE symbol IS NOT NULL
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()
    return {
        "documents": int(docs),
        "chunks": int(chunks),
        "by_symbol": {s: int(c) for s, c in by_sym},
        "vector_backend": cfg.vector_backend,
        "rerank_backend": cfg.rerank_backend,
        "retrieve_candidates": cfg.retrieve_candidates,
        "rerank_top_k": cfg.rerank_top_k,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finance F6 filings RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Load sample filings into pgvector")
    sub.add_parser("status", help="Document / chunk counts")
    p_list = sub.add_parser("list", help="List documents (optional symbol)")
    p_list.add_argument("symbol", nargs="?")
    p_search = sub.add_parser("search", help="Retrieve + dynamic rerank")
    p_search.add_argument("symbol")
    p_search.add_argument("query")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        print(json.dumps(ingest_sample(), indent=2))
        print(json.dumps({"status": status()}, indent=2))
        return 0
    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
        return 0
    if args.cmd == "list":
        print(json.dumps(list_documents(args.symbol), indent=2))
        return 0
    if args.cmd == "search":
        print(json.dumps(search_filings(args.query, symbol=args.symbol), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
