# -*- coding: utf-8 -*-
"""
akari-mem UserPromptSubmit hook for Claude Code.

Called on EVERY user message. Reads the user's query from stdin (JSON),
performs a lightweight semantic search against memory, and outputs
matching memories to stderr for Claude to use as context.

This is the "auto-memory recall" feature — Claude gets relevant memories
injected WITHOUT needing to explicitly call search_memory.

NOTE: This hook loads the FULL BGE-M3 model (~2.3GB) on every user message.
      For performance, consider using a lighter model or caching the model
      in a long-running process (like claude-mem's worker-service approach).
      For now, we use a lightweight keyword-only search (FTS5) to avoid
      the model loading overhead.
"""
import sys, os, json

_extra = os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.append(_extra)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HOME", "F:\\models")

try:
    # Read hook input from stdin
    raw = sys.stdin.read()
    hook_input = json.loads(raw) if raw.strip() else {}

    # Extract user's message
    user_query = hook_input.get("user_prompt", "")
    if not user_query or len(user_query) < 5:
        sys.exit(0)

    # Use lightweight keyword search (FTS5) — NO model loading!
    import sqlite3
    db_path = "F:/claude-tools/akari-mem-mcp/data/akari-mem.db"
    if not os.path.exists(db_path):
        sys.exit(0)

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Extract keywords from query (simple: split by space, take meaningful words)
    words = [w.strip() for w in user_query.split() if len(w.strip()) >= 2]
    if not words:
        sys.exit(0)

    # Build FTS5 OR query
    fts_query = " OR ".join(f'"{w}"' for w in words[:5])

    try:
        rows = db.execute(
            "SELECT m.id, m.title, m.tags FROM memories m "
            "JOIN memories_fts f ON m.id = f.rowid "
            "WHERE memories_fts MATCH ? "
            "ORDER BY rank LIMIT 5",
            (fts_query,),
        ).fetchall()
    except Exception:
        rows = []

    db.close()

    if rows:
        print(f"[akari-mem] Found {len(rows)} related memories:", file=sys.stderr)
        for r in rows:
            print(f"  #{r['id']} {r['title'][:60]}", file=sys.stderr)
        print(
            "[akari-mem] Use search_memory tool for full semantic search.",
            file=sys.stderr,
        )

except Exception as e:
    print(f"[akari-mem] Hook error: {e}", file=sys.stderr)
