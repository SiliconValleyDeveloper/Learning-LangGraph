"""Demo MCP server (stdio) — fake market tools for learning.

Run by the LangGraph client via MultiServerMCPClient, not by hand.

Tools:
  - list_tickers
  - get_quote
  - fx_rate

Uses the official MCP Python SDK (FastMCP).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-market")

# Deterministic fake quotes so the lesson works offline / without API keys.
_QUOTES: dict[str, dict[str, float | str]] = {
    "AAPL": {"price": 198.4, "currency": "USD", "name": "Apple Inc."},
    "MSFT": {"price": 425.1, "currency": "USD", "name": "Microsoft Corp."},
    "GOOGL": {"price": 175.2, "currency": "USD", "name": "Alphabet Inc."},
    "TSLA": {"price": 248.0, "currency": "USD", "name": "Tesla Inc."},
    "RELIANCE.NS": {"price": 2920.5, "currency": "INR", "name": "Reliance Industries"},
    "TCS.NS": {"price": 4150.0, "currency": "INR", "name": "Tata Consultancy Services"},
    "INFY.NS": {"price": 1880.0, "currency": "INR", "name": "Infosys Ltd"},
}

_FX: dict[str, float] = {
    "USDINR": 83.45,
    "EURUSD": 1.08,
    "GBPUSD": 1.27,
}


@mcp.tool()
def list_tickers() -> str:
    """List demo ticker symbols this MCP server understands."""
    lines = [f"{sym} — {meta['name']}" for sym, meta in _QUOTES.items()]
    return "Supported tickers:\n" + "\n".join(lines)


@mcp.tool()
def get_quote(symbol: str) -> str:
    """Return a demo stock quote for a ticker (e.g. AAPL, TCS.NS). Offline fake data."""
    key = (symbol or "").strip().upper()
    # Allow reliance.ns style input
    if key.endswith(".NS"):
        pass
    elif key in {"RELIANCE", "TCS", "INFY"}:
        key = f"{key}.NS"
    hit = _QUOTES.get(key)
    if not hit:
        known = ", ".join(sorted(_QUOTES))
        return f"Unknown symbol '{symbol}'. Try one of: {known}"
    return (
        f"{hit['name']} ({key}): {hit['price']} {hit['currency']} "
        f"(demo MCP quote — not live market data)"
    )


@mcp.tool()
def fx_rate(pair: str) -> str:
    """Return a demo FX rate for a pair like USDINR, EURUSD, GBPUSD."""
    key = (pair or "").strip().upper().replace("/", "")
    rate = _FX.get(key)
    if rate is None:
        return f"Unknown pair '{pair}'. Try: {', '.join(sorted(_FX))}"
    return f"{key} = {rate} (demo MCP rate — not live)"


if __name__ == "__main__":
    mcp.run(transport="stdio")
