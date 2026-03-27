"""
akari-mem-mcp — MCP memory server entry point.

Usage:
  python server.py                    # Start stdio MCP server
  python server.py --migrate          # Import from claude-mem first, then start
  python server.py --rebuild          # Rebuild vector index, then start
  python server.py --test             # Run quick search test
"""
import os

# ── CRITICAL: Silence model loading output BEFORE any imports ────
# MCP uses stdio pipes. Model loading (HF, tqdm, transformers) writes
# progress to stdout/stderr. If the pipe buffer fills and the host
# doesn't consume fast enough, the warmup thread blocks forever.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ.get("AKARI_MODEL_CACHE", ""))

import sys
import json
import asyncio
import logging

# Load .env and setup paths
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env_loader import setup, resolve_config, load_dotenv
setup()  # reads .env, may append AKARI_MEM_LIBS to sys.path

# ── CRITICAL: Remove polluting global Python 3.14 paths from sys.path ──
# env_loader.setup() reads AKARI_MEM_LIBS=F:\python-libs from .env and appends it.
# F:\python-libs pulls in C:\Python314's torchvision which has a circular import
# bug that crashes sentence_transformers loading.
# Use BLOCKLIST (remove known bad paths) not allowlist — preserves Python stdlib.
_BAD_PATH_MARKERS = [
    os.path.normcase(r"F:\python-libs"),
    os.path.normcase(r"C:\Python314"),
]
sys.path[:] = [
    _p for _p in sys.path
    if not any(
        os.path.normcase(os.path.abspath(_p)).startswith(_bad)
        for _bad in _BAD_PATH_MARKERS
    )
]
os.environ.pop("PYTHONPATH", None)
os.environ.pop("AKARI_MEM_LIBS", None)  # prevent future re-injection

from mcp.server.fastmcp import FastMCP
from store import MemoryStore
from embeddings import create_provider
from rerank import create_reranker

# Pre-import heavy libs at module level (before MCP stdio takes over).
# These imports trigger grpc/protobuf/tqdm/CUDA init that writes to
# stdout/stderr. If done inside the warmup thread (after MCP starts),
# the output fills the pipe buffer and blocks the thread forever.
try:
    import chromadb  # noqa: F401 — grpc/protobuf init writes to stderr
except ImportError:
    pass
try:
    import sentence_transformers  # noqa: F401 — tqdm/transformers logging
except ImportError:
    pass
try:
    import torch  # noqa: F401 — CUDA init output
except ImportError:
    pass

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

# ── Store + Background Index Queue ──────────────────────────
import threading
import queue as _queue_mod

_store = None
_store_lock = threading.Lock()
_warmup_state = "idle"   # idle → loading → ready | failed
_warmup_error = None
_index_queue = _queue_mod.Queue()  # items: (mem_id, title, text, tags, project, source)

def get_store():
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:
            p = create_provider(config.get("embedding", {}))
            r = create_reranker(config.get("rerank", {}))
            _store = MemoryStore(data_dir=config["data_dir"], embedding_provider=p, reranker=r)
    return _store

def _background_warmup():
    """Pre-load models in background so first tool call is fast."""
    global _warmup_state, _warmup_error
    import time
    time.sleep(0.5)  # let MCP handshake complete first
    _warmup_state = "loading"

    # File-based diagnostic log (bypasses MCP stdio pipe)
    _log_path = os.path.join(config["data_dir"], "warmup.log")
    def _log(msg):
        try:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                f.flush()
        except Exception:
            pass

    try:
        _log("Warmup started")
        _log("Step 1: create_provider...")
        p = create_provider(config.get("embedding", {}))
        _log(f"Step 1 done: provider={type(p).__name__}")

        _log("Step 2: create_reranker...")
        r = create_reranker(config.get("rerank", {}))
        _log(f"Step 2 done: reranker={type(r).__name__}")

        _log("Step 3: MemoryStore init...")
        from store import MemoryStore as _MS
        _store_obj = _MS(data_dir=config["data_dir"], embedding_provider=p, reranker=r)
        _log("Step 3 done: MemoryStore ready")

        global _store
        _store = _store_obj
        _warmup_state = "ready"
        _log("Warmup COMPLETE - models loaded")
    except Exception as e:
        _warmup_state = "failed"
        _warmup_error = str(e)
        import traceback
        _log(f"Warmup FAILED: {e}\n{traceback.format_exc()}")

def _save_to_sqlite(title, text, tags, project, source):
    """Fast SQLite-only save. Returns mem_id."""
    import sqlite3
    from datetime import datetime, timezone
    db_path = os.path.join(config["data_dir"], "akari-mem.db")
    now = datetime.now(timezone.utc).isoformat()
    db = sqlite3.connect(db_path)
    cur = db.execute(
        "INSERT INTO memories (title, text, tags, project, source, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, text, tags, project, source, now, now),
    )
    mem_id = cur.lastrowid
    db.commit()
    db.close()
    return mem_id

def _index_worker():
    """Background thread: picks up saved memories and indexes them in ChromaDB."""
    while True:
        item = _index_queue.get()
        if item is None:
            break  # shutdown signal
        mem_id, title, text, tags, project, source = item
        try:
            store = get_store()
            document = f"{title}\n{text}"
            store._collection.add(
                ids=[f"mem_{mem_id}"],
                documents=[document],
                metadatas=[{
                    "sqlite_id": mem_id,
                    "title": title[:200],
                    "tags": tags,
                    "project": project,
                    "source": source,
                }],
            )
            logger.info(f"Indexed memory #{mem_id} in ChromaDB.")
        except Exception as e:
            logger.exception(f"Failed to index memory #{mem_id}: {e}")
        _index_queue.task_done()

# ── MCP Server ──────────────────────────────────────────────

mcp = FastMCP(
    "akari-mem",
    instructions=(
        "Akari's personal memory system with dual search strategy.\n"
        "- quick_search: instant keyword search (FTS5), use for simple lookups, name/term matching\n"
        "- search_memory: deep hybrid search (vector+keyword+RRF+rerank), use for semantic/complex queries\n"
        "- get_memory(id): fetch full content of a memory by ID (use after search)\n"
        "- save_memory: store important findings\n"
        "- list_memories: see recent entries\n"
        "Prefer quick_search first; escalate to search_memory if results are insufficient.\n"
        "Search returns compact summaries — use get_memory(id) when you need the full text."
    ),
)


@mcp.tool()
async def quick_search(query: str, limit: int = 5) -> str:
    """
    Fast keyword search (FTS5). Millisecond-level, no model loading.
    Best for: exact name matching, specific terms, simple lookups.
    Use this FIRST. Escalate to search_memory if results are poor.

    Args:
        query: Search keywords (space-separated, OR logic)
        limit: Max results (default 5)
    """
    def _impl():
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
            fts_q = " OR ".join(f'"{ t}"' for t in terms)
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
            preview = r["text"][:100].replace("\n", " ")
            tags_str = f"  Tags: {r['tags']}" if r["tags"] else ""
            lines.append(f"#{r['id']} | {r['title']} | {preview}...{tags_str}")
        return "\n".join(lines)

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def search_memory(query: str, limit: int = 5) -> str:
    """
    Deep hybrid search: vector(BGE-M3) + keyword(FTS5) + RRF fusion + rerank.
    Takes 5-15s on first call (model loading), ~1-3s after warm-up.
    Best for: semantic/conceptual queries, fuzzy matching, complex questions.

    Args:
        query: Natural language search query
        limit: Max results (default 5)
    """
    def _impl():
        results = get_store().search(query, limit)
        if not results:
            return "No memories found."

        lines = [f"[deep] Found {len(results)} memories (use get_memory(id) for full content):\n"]
        for r in results:
            rrf = r.get("rrf_score", "-")
            preview = r["text"][:100].replace("\n", " ")
            tags_str = f"  tags={r['tags']}" if r.get("tags") else ""
            proj_str = f"  proj={r['project']}" if r.get("project") else ""
            lines.append(f"#{r['id']} [rrf={rrf}] {r['title']}{tags_str}{proj_str}")
            lines.append(f"  {preview}...")
        return "\n".join(lines)

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def get_memory(memory_id: int) -> str:
    """
    Get full content of a memory by ID. Use after search to read complete details.

    Args:
        memory_id: The ID of the memory to retrieve
    """
    def _impl():
        import sqlite3
        db_path = os.path.join(config["data_dir"], "akari-mem.db")
        if not os.path.exists(db_path):
            return "No memory database found."
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        db.close()
        if not row:
            return f"Memory #{memory_id} not found."

        lines = [
            f"=== Memory #{row['id']}: {row['title']} ===",
            row["text"],
            "",
        ]
        if row["tags"]:
            lines.append(f"Tags: {row['tags']}")
        if row["project"]:
            lines.append(f"Project: {row['project']}")
        if row["source"]:
            lines.append(f"Source: {row['source']}")
        lines.append(f"Created: {row['created_at']}")
        return "\n".join(lines)

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def save_memory(title: str, text: str, tags: str = "", project: str = "") -> str:
    """
    Save a new memory. Automatically embedded for future search.

    Args:
        title: Short descriptive title
        text: Full memory content
        tags: Comma-separated tags (optional)
        project: Project name (optional)
    """
    # Fast path: write to SQLite immediately (< 10ms)
    mem_id = await asyncio.to_thread(
        _save_to_sqlite, title, text, tags, project, "mcp"
    )
    # Queue background indexing (ChromaDB embedding) — non-blocking
    _index_queue.put((mem_id, title, text, tags, project, "mcp"))
    pending = _index_queue.qsize()
    status = f"Memory saved: #{mem_id} — {title}"
    if _warmup_state != "ready":
        status += f" (vector indexing queued, models {_warmup_state})"
    elif pending > 1:
        status += f" ({pending} pending index)"
    return status


@mcp.tool()
async def list_memories(limit: int = 10) -> str:
    """
    List the most recent memories.

    Args:
        limit: Number of memories to show (default 10)
    """
    def _impl():
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

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def delete_memory(memory_id: int) -> str:
    """
    Delete a memory by ID.

    Args:
        memory_id: The ID of the memory to delete
    """
    def _impl():
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

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def get_stats() -> str:
    """Get memory store statistics: total count, embedding model, projects, etc."""
    def _impl():
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
            f"  Warmup: {_warmup_state}" + (f" (error: {_warmup_error})" if _warmup_error else ""),
            f"  Index queue: {_index_queue.qsize()} pending",
        ]
        if projects:
            lines.append("  Projects:")
            for proj, cnt in projects.items():
                lines.append(f"    {proj}: {cnt}")
        return "\n".join(lines)

    return await asyncio.to_thread(_impl)


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

    # Start background workers
    threading.Thread(target=_background_warmup, daemon=True, name="warmup").start()
    threading.Thread(target=_index_worker, daemon=True, name="indexer").start()

    # Start MCP stdio server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
