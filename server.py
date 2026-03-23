"""
akari-mem-mcp — MCP memory server entry point.

Usage:
  python server.py                    # Start stdio MCP server
  python server.py --migrate          # Import from claude-mem first, then start
  python server.py --rebuild          # Rebuild vector index, then start
  python server.py --test             # Run quick search test
"""
import sys
import os
import json
import asyncio
import logging

# Load .env and setup paths
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env_loader import setup, resolve_config, load_dotenv
setup()

from mcp.server.fastmcp import FastMCP
from store import MemoryStore
from embeddings import create_provider
from rerank import create_reranker

# ── Config ──────────────────────────────────────────────────

CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")

def load_config() -> dict:
    """Load config from config.json + env vars."""
    defaults = {
        "data_dir": os.path.join(_PROJECT_ROOT, "data"),
        "embedding": {"mode": "default"},
        "log_level": "INFO",
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        defaults.update(user_cfg)
    return resolve_config(defaults)


# ── Globals ─────────────────────────────────────────────────

config = load_config()

logging.basicConfig(
    level=getattr(logging, config.get("log_level", "INFO")),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("akari-mem")

# Lazy store — models load on FIRST USE, not at startup (avoids MCP handshake timeout)
_store = None

def get_store():
    global _store
    if _store is None:
        logger.info("Loading models (first use)...")
        p = create_provider(config.get("embedding", {}))
        r = create_reranker(config.get("rerank", {}))
        _store = MemoryStore(data_dir=config["data_dir"], embedding_provider=p, reranker=r)
        logger.info("MemoryStore ready.")
    return _store

# ── MCP Server ──────────────────────────────────────────────

mcp = FastMCP(
    "akari-mem",
    instructions=(
        "Akari's personal memory system with dual search strategy.\n"
        "- quick_search: instant keyword search (FTS5), use for simple lookups, name/term matching\n"
        "- search_memory: deep hybrid search (vector+keyword+RRF+rerank), use for semantic/complex queries\n"
        "- save_memory: store important findings\n"
        "- list_memories: see recent entries\n"
        "Prefer quick_search first; escalate to search_memory if results are insufficient."
    ),
)


@mcp.tool()
def quick_search(query: str, limit: int = 5) -> str:
    """
    Fast keyword search (FTS5). Millisecond-level, no model loading.
    Best for: exact name matching, specific terms, simple lookups.
    Use this FIRST. Escalate to search_memory if results are poor.

    Args:
        query: Search keywords (space-separated, OR logic)
        limit: Max results (default 5)
    """
    import sqlite3
    db_path = os.path.join(config["data_dir"], "akari-mem.db")
    if not os.path.exists(db_path):
        return "No memory database found."

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            "SELECT m.* FROM memories m "
            "JOIN memories_fts f ON m.id = f.rowid "
            "WHERE memories_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
    except Exception:
        # FTS5 syntax error — split into OR terms
        terms = [t.strip() for t in query.split() if t.strip()]
        fts_q = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = db.execute(
                "SELECT m.* FROM memories m "
                "JOIN memories_fts f ON m.id = f.rowid "
                "WHERE memories_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_q, limit),
            ).fetchall()
        except Exception:
            rows = []
    db.close()

    if not rows:
        return "No keyword matches. Try search_memory for semantic search."

    lines = [f"[quick] Found {len(rows)} matches:\n"]
    for r in rows:
        lines.append(f"--- [#{r['id']}] {r['title']} ---")
        lines.append(r["text"][:300])
        if r["tags"]:
            lines.append(f"  Tags: {r['tags']}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """
    Deep hybrid search: vector(BGE-M3) + keyword(FTS5) + RRF fusion + rerank.
    Takes 5-15s on first call (model loading), ~1-3s after warm-up.
    Best for: semantic/conceptual queries, fuzzy matching, complex questions.

    Args:
        query: Natural language search query
        limit: Max results (default 5)
    """
    results = get_store().search(query, limit)
    if not results:
        return "No memories found."

    lines = [f"[deep] Found {len(results)} memories:\n"]
    for r in results:
        dist = r.get("distance", "-")
        rrf = r.get("rrf_score", "-")
        lines.append(f"--- [#{r['id']}] {r['title']} (dist={dist} rrf={rrf}) ---")
        lines.append(r["text"][:300])
        if r.get("tags"):
            lines.append(f"  Tags: {r['tags']}")
        if r.get("project"):
            lines.append(f"  Project: {r['project']}")
        lines.append(f"  Created: {r['created_at']}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def save_memory(title: str, text: str, tags: str = "", project: str = "") -> str:
    """
    Save a new memory. Automatically embedded for future search.

    Args:
        title: Short descriptive title
        text: Full memory content
        tags: Comma-separated tags (optional)
        project: Project name (optional)
    """
    mem_id = get_store().save(title, text, tags=tags, project=project, source="mcp")
    return f"Memory saved: #{mem_id} — {title}"


@mcp.tool()
def list_memories(limit: int = 10) -> str:
    """
    List the most recent memories.

    Args:
        limit: Number of memories to show (default 10)
    """
    import sqlite3
    db_path = os.path.join(config["data_dir"], "akari-mem.db")
    if not os.path.exists(db_path):
        return "No memories stored yet."
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    db.close()
    if not rows:
        return "No memories stored yet."

    lines = [f"Latest {len(rows)} memories:\n"]
    for r in rows:
        tags_str = f" [{r['tags']}]" if r["tags"] else ""
        lines.append(f"  #{r['id']}: {r['title']}{tags_str} ({r['created_at'][:10]})")
    return "\n".join(lines)


@mcp.tool()
def delete_memory(memory_id: int) -> str:
    """
    Delete a memory by ID.

    Args:
        memory_id: The ID of the memory to delete
    """
    import sqlite3
    db_path = os.path.join(config["data_dir"], "akari-mem.db")
    db = sqlite3.connect(db_path)
    cur = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    deleted = cur.rowcount > 0
    db.close()
    # Also remove from ChromaDB if store is loaded
    if deleted and _store is not None:
        try:
            _store._collection.delete(where={"sqlite_id": memory_id})
        except Exception:
            pass
    if deleted:
        return f"Memory #{memory_id} deleted."
    return f"Memory #{memory_id} not found."


@mcp.tool()
def get_stats() -> str:
    """Get memory store statistics: total count, embedding model, projects, etc."""
    import sqlite3
    db_path = os.path.join(config["data_dir"], "akari-mem.db")
    if not os.path.exists(db_path):
        return "No memory database found."
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    total = db.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()["cnt"]
    latest = db.execute("SELECT title FROM memories ORDER BY id DESC LIMIT 1").fetchone()
    projects = {}
    for row in db.execute("SELECT project, COUNT(*) as cnt FROM memories WHERE project != '' GROUP BY project"):
        projects[row["project"]] = row["cnt"]
    db.close()

    emb_cfg = config.get("embedding", {})
    rnk_cfg = config.get("rerank", {})
    lines = [
        "=== Akari Memory Stats ===",
        f"  Total memories: {total}",
        f"  Embedding: {emb_cfg.get('model', emb_cfg.get('mode', 'default'))}",
        f"  Rerank: {rnk_cfg.get('model', rnk_cfg.get('mode', 'none'))}",
        f"  Latest: {latest['title'] if latest else 'none'}",
        f"  Data dir: {config['data_dir']}",
        f"  Models loaded: {'yes' if _store else 'no (lazy)'}",
    ]
    if projects:
        lines.append("  Projects:")
        for proj, cnt in projects.items():
            lines.append(f"    {proj}: {cnt}")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────

def run_migrate():
    """Import from claude-mem."""
    from migrate import migrate_from_claude_mem
    migrate_from_claude_mem(get_store())


def run_test():
    """Quick search test."""
    queries = [
        "阿卡林的猫叫什么名字",
        "赋驾公司的老板",
        "TPM 职业转型",
        "RTX 5070 调优",
    ]
    s = get_store()
    print(f"Store stats: {json.dumps(s.get_stats(), ensure_ascii=False, indent=2)}\n")
    for q in queries:
        results = s.search(q, 3)
        print(f'Query: "{q}"')
        for r in results:
            print(f"  #{r['id']} [{r['distance']}] {r['title'][:50]}")
        print()


def main():
    if "--migrate" in sys.argv:
        run_migrate()
    if "--rebuild" in sys.argv:
        get_store().rebuild_vectors()
    if "--test" in sys.argv:
        run_test()
        return

    # Start MCP stdio server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
