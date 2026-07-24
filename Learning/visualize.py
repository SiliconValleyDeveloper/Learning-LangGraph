"""Print a compiled LangGraph using built-in visualizers (no extension needed).

Requires: grandalf  (for ASCII) — already in requirements.txt

Usage:
    from visualize import show_graph
    show_graph(graph)
"""


def show_graph(graph, title: str = "Graph structure") -> None:
    """Show ASCII diagram + Mermaid source for a compiled graph."""
    g = graph.get_graph()

    print(f"\n=== {title} (ASCII) ===\n")
    try:
        print(g.draw_ascii())
    except ImportError as exc:
        print(f"  (ASCII skipped: {exc})")
        print("  Fix: pip install grandalf")

    print(f"\n=== {title} (Mermaid) ===")
    print("Paste into https://mermaid.live for a nicer picture:\n")
    print(g.draw_mermaid())
    print()
