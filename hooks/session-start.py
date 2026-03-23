# -*- coding: utf-8 -*-
"""
akari-mem SessionStart hook for Claude Code.

Called by Claude Code's SessionStart hook to auto-inject recent memories.
Outputs to stderr which Claude Code reads.

Setup: Add to hooks.json SessionStart array.
"""
import sys, os, json

# Setup paths via env_loader
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from env_loader import setup, resolve_config
setup()

try:
    from store import MemoryStore
    from embeddings import create_provider

    # Load config
    config_path = os.path.join(_project_root, "config.json")
    defaults = {"data_dir": os.path.join(_project_root, "data"), "embedding": {"mode": "default"}}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            defaults.update(json.load(f))
    config = resolve_config(defaults)

    provider = create_provider(config.get("embedding", {}))
    store = MemoryStore(data_dir=config["data_dir"], embedding_provider=provider)

    # Get recent memories (last 10)
    recent = store.list_recent(10)

    if recent:
        lines = []
        lines.append("[akari-mem] Auto-loaded recent memories:")
        lines.append("")
        for m in recent:
            tags = m.get("tags", "") or ""
            lines.append(f"  #{m['id']} {m['title'][:60]}  [{tags[:30]}]")
        lines.append("")
        lines.append(f"[akari-mem] Total: {len(recent)} recent | Use search_memory for semantic search")
        lines.append("")

        for line in lines:
            print(line, file=sys.stderr)

except Exception as e:
    print(f"[akari-mem] Hook error: {e}", file=sys.stderr)
