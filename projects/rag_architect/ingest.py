"""Load Contoso Ops KB markdown and build dense + BM25 indexes."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from threading import Lock

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from projects.rag_architect.config import RagArchitectConfig, load_config

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_lock = Lock()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._docs_tokens = [tokenize(d.page_content) for d in documents]
        self.N = len(documents)
        self.avgdl = (
            sum(len(t) for t in self._docs_tokens) / self.N if self.N else 0.0
        )
        df: Counter[str] = Counter()
        for toks in self._docs_tokens:
            df.update(set(toks))
        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.documents:
            return []
        scored: list[tuple[Document, float]] = []
        for doc, toks in zip(self.documents, self._docs_tokens, strict=True):
            if not toks:
                continue
            tf = Counter(toks)
            score = 0.0
            dl = len(toks)
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1) / denom)
            if score > 0:
                scored.append((doc, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


class KnowledgeIndex:
    """In-memory dense vector store + BM25 over the same chunks."""

    def __init__(self, config: RagArchitectConfig | None = None) -> None:
        self.config = config or load_config()
        self.documents: list[Document] = []
        self.chunks: list[Document] = []
        self.vector: InMemoryVectorStore | None = None
        self.bm25: BM25Index | None = None
        self._embeddings = OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True,
        )

    def load_markdown(self, data_dir: Path | None = None) -> list[Document]:
        root = data_dir or self.config.data_dir
        docs: list[Document] = []
        for path in sorted(Path(root).glob("*.md")):
            docs.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name, "visibility": "internal"},
                )
            )
        if not docs:
            raise RuntimeError(f"No markdown documents in {root}")
        self.documents = docs
        return docs

    def build(self, data_dir: Path | None = None) -> KnowledgeIndex:
        docs = self.load_markdown(data_dir)
        chunks = self._splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata.get('source', 'doc')}::chunk::{i}"
            )
        self.chunks = chunks
        store = InMemoryVectorStore(embedding=self._embeddings)
        store.add_documents(chunks)
        self.vector = store
        self.bm25 = BM25Index(chunks)
        return self

    @property
    def ready(self) -> bool:
        return bool(self.chunks and self.vector and self.bm25)


_INDEX: KnowledgeIndex | None = None


def get_index(*, rebuild: bool = False) -> KnowledgeIndex:
    global _INDEX
    with _lock:
        if _INDEX is None or rebuild or not _INDEX.ready:
            _INDEX = KnowledgeIndex().build()
        return _INDEX


def ingest_seed(*, rebuild: bool = True) -> dict[str, int | str]:
    index = get_index(rebuild=rebuild)
    return {
        "documents": len(index.documents),
        "chunks": len(index.chunks),
        "data_dir": str(index.config.data_dir),
    }
