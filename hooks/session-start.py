# -*- coding: utf-8 -*-
"""
akari-mem SessionStart hook for Claude Code.

Called by Claude Code's SessionStart hook to auto-inject recent memories
into the conversation context. Outputs to stderr which Claude Code reads.

Usage in hooks.json:
  "SessionStart": [{
    "matcher": "*",
    "hooks": [{
      "type": "command",
      "command": "python F:\\claude-tools\\akari-mem-mcp\\hooks\\session-start.py",
      "timeout": 30
    }]
  }]
"""
import sys, os, json

# Ensure correct paths
_extra = os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.append(_extra)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HOME", "F:\\models")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from store import MemoryStore
    from embeddings import create_provider

    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    provider = create_provider(config.get("embedding", {}))
    store = MemoryStore(
        data_dir=config["data_dir"], embedding_provider=provider
    )

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

        # Output to stderr — Claude Code hooks capture stderr
        for line in lines:
            print(line, file=sys.stderr)

except Exception as e:
    print(f"[akari-mem] Hook error: {e}", file=sys.stderr)
