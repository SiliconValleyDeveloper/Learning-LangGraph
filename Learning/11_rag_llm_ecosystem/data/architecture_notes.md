# Architecture Notes

## Indexing path

documents → chunks → embeddings → vector store

## Query path

question → query embedding → top-k retrieval → prompt with context → LLM answer

## Component roles

- LangChain: document loaders, splitters, embeddings, retrievers, prompts
- LangGraph: explicit retrieve and generate nodes with shared state
- Ollama: local chat and embedding models
- Angular lab: visualize stages, sources, and retrieved chunks

## Design rules

Prefer small chunks with overlap for precise retrieval. Keep source filenames in
metadata. Instruct the model to cite sources and admit missing evidence. Private
files should remain local-only and marked with `visibility=private` metadata.
