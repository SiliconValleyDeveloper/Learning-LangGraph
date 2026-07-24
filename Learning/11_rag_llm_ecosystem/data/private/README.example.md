# Private document template

Copy this file to a new name inside `data/private/` for local-only RAG content.

```text
cp data/private/README.example.md data/private/my_private_notes.md
```

Private files are:

- loaded into the local vector store
- labeled with `visibility=private`
- ignored by git so they are not pushed to GitHub
