"""
Migrate memories from claude-mem into akari-mem-mcp.

Reads all observations from ~/.claude-mem/claude-mem.db
and imports them into the new store with proper deduplication.
"""
import sqlite3
import os
import logging

logger = logging.getLogger("akari-mem.migrate")

CLAUDE_MEM_DB = os.path.expanduser("~/.claude-mem/claude-mem.db").replace("\\", "/")


def migrate_from_claude_mem(store, db_path: str = CLAUDE_MEM_DB):
    """Import observations from claude-mem into akari-mem store."""
    if not os.path.exists(db_path):
        logger.error(f"claude-mem DB not found: {db_path}")
        print(f"ERROR: {db_path} not found")
        return

    # Read all observations
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT id, memory_session_id, project, type, title, text, created_at "
        "FROM observations ORDER BY id"
    ).fetchall()
    db.close()

    if not rows:
        print("No observations found in claude-mem.")
        return

    # Check for existing imports to prevent duplicates
    existing = store.list_recent(9999)
    existing_titles = {m["title"] for m in existing}

    imported = 0
    skipped = 0

    for row in rows:
        title = row["title"] or f"claude-mem #{row['id']}"
        text = row["text"] or ""

        if title in existing_titles:
            logger.debug(f"SKIP (duplicate): {title[:40]}")
            skipped += 1
            continue

        # Map claude-mem fields to akari-mem
        tags = row["type"] or ""
        project = row["project"] or ""
        source = f"claude-mem:{row['memory_session_id']}"

        store.save(
            title=title,
            text=text,
            tags=tags,
            project=project,
            source=source,
        )
        imported += 1
        print(f"  [+] #{row['id']}: {title[:50]}")

    print(f"\nMigration complete: {imported} imported, {skipped} skipped (duplicates)")
    print(f"Total in akari-mem: {store.get_stats()['total_memories']}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from store import MemoryStore
    from embeddings import create_provider
    import json

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    provider = create_provider(config.get("embedding", {}))
    data_dir = config.get("data_dir", os.path.join(os.path.dirname(__file__), "data"))
    s = MemoryStore(data_dir=data_dir, embedding_provider=provider)
    migrate_from_claude_mem(s)
