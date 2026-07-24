# Product FAQ

## What is LangGraph Lab?

LangGraph Lab is a local visual console for learning stateful agent graphs.
Learners can switch concepts, inspect topology, and watch nodes light up.

## What is RAG in this project?

RAG retrieves relevant chunks from local markdown files, then asks the chat
model to answer only from that evidence. The answer should cite source files.

## Which models are required?

- Chat: `qwen3:8b`
- Embeddings: `nomic-embed-text`

Both run through Ollama on the learner machine.

## Can private documents be used?

Yes. Place private markdown files in `Learning/11_rag_llm_ecosystem/data/private/`.
They are loaded for local retrieval but ignored by git so secrets stay off the
remote repository.
