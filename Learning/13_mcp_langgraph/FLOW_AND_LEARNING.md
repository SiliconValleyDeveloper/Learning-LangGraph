# Phase 13 · MCP + LangGraph + LLM

**Goal:** use **Model Context Protocol (MCP)** tools inside a **LangGraph** ReAct agent powered by your local **Ollama** LLM.

---

## 1. Big picture (layman)

```text
MCP server  = tool shop (get_quote, fx_rate, …)
MCP client  = shopper in your Python app
Adapter     = translates shop tools into LangChain tools
LangGraph   = agent loop (LLM decides when to buy/use a tool)
LLM         = brain that only sees tool names + schemas
```

| Piece | In this phase |
|-------|----------------|
| MCP server | `demo_mcp_server.py` (stdio) or Yahoo via `npx` |
| Client + adapter | `langchain-mcp-adapters` → `MultiServerMCPClient` |
| Graph | Same as Phase 3: `agent` ⇄ `tools` |
| LLM | `ChatOllama` via `Learning/llm.py` |

**Why MCP?** Write tools once; reuse them in **Cursor**, **LangGraph agents**, and other MCP clients without rewriting glue code.

---

## 2. Why these libraries?

| Library | Job | Why not DIY? |
|---------|-----|--------------|
| **`mcp` (FastMCP)** | Define + serve tools over stdio/HTTP | Protocol handshake, schemas, transports are already specified |
| **`langchain-mcp-adapters`** | `get_tools()` → LangChain `BaseTool` list | Avoid hand-wrapping every MCP tool for `bind_tools` / `ToolNode` |
| **LangGraph** | Orchestrate agent ↔ tools | Conditional edges + state you already know from Phase 3 |
| **Ollama / ChatOllama** | Local LLM with tool calling | No cloud API key for the agent brain |

---

## 3. Lessons

### Lesson 1 — local demo MCP (no API key)

```bash
source .venv/bin/activate
pip install langchain-mcp-adapters mcp   # already in requirements.txt
python Learning/13_mcp_langgraph/01_mcp_tools_agent.py
```

Flow:

```text
01_mcp_tools_agent.py
   └─ MultiServerMCPClient spawns demo_mcp_server.py (stdio)
         └─ tools: list_tickers, get_quote, fx_rate
               └─ LangGraph ReAct with Ollama
```

Try:

```bash
python Learning/13_mcp_langgraph/01_mcp_tools_agent.py "Demo quote for AAPL and USDINR"
```

### Lesson 2 — optional live Yahoo MCP (same as Cursor)

Uses your Cursor config pattern:

```json
"yahoo-finance": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-yahoo-finance@latest"]
}
```

```bash
python Learning/13_mcp_langgraph/02_mcp_yahoo_optional.py
```

Needs **Node/npx** + **network**. First run may download the package.

---

## 4. Code path (Lesson 1)

1. Start MCP server as a child process (`command` + `args` + `transport: stdio`)
2. `tools = await client.get_tools()`
3. `llm.bind_tools(tools)`
4. Graph: `START → agent → tools_condition → tools → agent → … → END`

That is **exactly** Phase 3 ReAct — only the tool source changed (MCP instead of `@tool` in the same file).

---

## 5. Interview answers

**Q: What is MCP?**  
A standard way for apps to expose tools/resources to LLMs. Servers provide tools; clients call them.

**Q: How does MCP plug into LangGraph?**  
Adapters convert MCP tools into LangChain tools → `bind_tools` + `ToolNode` / `create_react_agent`.

**Q: MCP vs plain `@tool`?**  
`@tool` = local to one app. MCP = shareable across Cursor, agents, services.

**Q: stdio vs HTTP?**  
stdio = spawn a local process (great for demos / Cursor). HTTP = remote/shared servers (better for production APIs).

---

## 6. File map

```text
Learning/13_mcp_langgraph/
  FLOW_AND_LEARNING.md      # this doc
  demo_mcp_server.py        # FastMCP stdio server (fake quotes)
  01_mcp_tools_agent.py     # LangGraph + demo MCP + Ollama
  02_mcp_yahoo_optional.py  # LangGraph + Yahoo MCP (live)
```

---

## 7. Visual lab

```bash
uvicorn api.main:app --reload --port 8000
cd ui && npm start
```

- Chip: **MCP · LangGraph · LLM**
- Direct URL: http://localhost:4200/chat/mcp
- Topbar: **MCP Agent**

Ask for a demo quote (e.g. TCS.NS) and watch **agent ⇄ MCP tools** light up, plus the MCP tool trace under the answer.

---

## 8. Next ideas

- Add FinStack / India-stock MCP the same way as Yahoo (`npx` from `~/.cursor/mcp.json`)
- Mount MCP tools on `advanced_chatbot` as an optional “markets” intent
- Prefer HTTP MCP when embedding agents inside FastAPI (stdio in web workers is awkward)
