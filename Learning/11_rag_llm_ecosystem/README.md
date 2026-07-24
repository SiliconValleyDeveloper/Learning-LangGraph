# Phase 11 — RAG + Lang Ecosystem + LLM

This phase builds one idea at a time:

1. **LLM** — a model predicts text from messages.
2. **LangChain** — reusable prompts, models, documents, embeddings, and retrievers.
3. **LangGraph** — explicit stateful workflows around those components.
4. **RAG** — retrieve relevant document chunks, then give them to the LLM as context.
5. **LangSmith** — traces and evaluates runs; it is discussed here but is not required.

## Prerequisites

From the repository root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## Run in order

```bash
python Learning/11_rag_llm_ecosystem/01_llm_basics.py
python Learning/11_rag_llm_ecosystem/02_lang_ecosystem.py
python Learning/11_rag_llm_ecosystem/03_chunk_embed.py
python Learning/11_rag_llm_ecosystem/04_retrieve.py
python Learning/11_rag_llm_ecosystem/05_rag_graph.py
python Learning/11_rag_llm_ecosystem/06_complex_rag_graph.py
```

## Lessons

| File | Level | What it teaches |
|------|-------|-----------------|
| `01`–`04` | Basics | LLM, Lang stack, chunk/embed, retrieve |
| `05_rag_graph.py` | Simple | `retrieve → generate` |
| `06_complex_rag_graph.py` | Complex | classify → rewrite → retrieve → grade → retry → generate → verify |

### Complex RAG control loop

```text
START → classify → rewrite → retrieve → grade
                                      ├─ pass ──────────────→ generate → verify → END
                                      └─ fail & retries left → bump_retry → rewrite …
                                      └─ fail & no retries  → generate (refuse/weak) → verify → END
```

Why this matters for real projects:

- **classify** — public vs private document access
- **rewrite** — better search queries than raw user text
- **grade + retry** — do not answer from bad evidence
- **verify** — enforce citations before trusting the answer

In the UI, open concept **Complex RAG · rewrite · grade · retry** (`rag_complex`).

## RAG data flow

```text
documents → chunks → embeddings → vector store
                                      ↓
question → rewrite → retrieve → grade → (retry?) → generate → verify → answer
```

An embedding is a numeric representation of meaning. Similar text tends to have
nearby vectors. RAG does not train the LLM — it supplies context at request time.

## Lang ecosystem map

| Library | Responsibility |
|---|---|
| LangChain | Documents, splitters, embeddings, retrievers, prompts |
| LangGraph | Stateful workflows, branches, loops, retries |
| Ollama | Local chat + embedding models |

After lessons 5–6, start the API/UI and inspect the RAG concepts in the lab.

## Knowledge files

Public documents live in `data/*.md` and are committed to the repo.

Private documents live in `data/private/*.md`. They are loaded for local RAG,
labeled `visibility=private`, and ignored by git (except `README.example.md`).

```bash
# add your own private notes
cp Learning/11_rag_llm_ecosystem/data/private/README.example.md \
  Learning/11_rag_llm_ecosystem/data/private/my_notes.md
```

Restart the API after adding files so the vector store reloads.
