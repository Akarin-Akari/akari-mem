# -*- coding: utf-8 -*-
"""
akari-mem UserPromptSubmit hook for Claude Code.

Called on every user message. Performs lightweight FTS5 keyword search
and outputs matching memories to stderr for Claude to use as context.

No model loading — pure SQLite FTS5 for instant results.
"""
import sys, os, json, sqlite3

# Setup paths via env_loader
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from env_loader import setup, resolve_config
setup()

# Resolve data dir from env
defaults = {"data_dir": os.path.join(_project_root, "data")}
config_path = os.path.join(_project_root, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        defaults.update(json.load(f))
config = resolve_config(defaults)
db_path = os.path.join(config["data_dir"], "akari-mem.db")

try:
    # Read hook input from stdin
    raw = sys.stdin.read()
    hook_input = json.loads(raw) if raw.strip() else {}

    # Extract user's message
    user_query = hook_input.get("user_prompt", "")
    if not user_query or len(user_query) < 5:
        sys.exit(0)

    if not os.path.exists(db_path):
        sys.exit(0)

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Extract keywords (split by space, take meaningful words)
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
