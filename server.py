"""
akari-mem-mcp — MCP memory server entry point (THIN CLIENT 版本).

ROLE (改造后职责):
=================
- 纯 MCP stdio 协议入口 (每个 Claude Code agent 一个独立进程)
- 不加载任何模型, 不持有 store
- 所有数据操作通过 HTTP 转发给 daemon (api_server.py)
- 多个 agent 共享同一个 daemon → 模型只加载一次, VRAM 共享

Usage:
  python server.py                    # MCP stdio thin client (默认)
  python server.py --migrate          # 本地迁移 (一次性, 临时加载 store)
  python server.py --rebuild          # 重建向量索引 (一次性, 临时加载)
  python server.py --rebuild-fts      # 重建 FTS5 索引 (一次性)
  python server.py --test             # 本地快速测试
"""
import os
import sys

# ──────────────────────────────────────────────────────────────
# CRITICAL: Redirect OS-level stderr (fd 2) to a file BEFORE imports
# 即使是 thin client, MCP stdio 管道阻塞规则仍然适用 (mcp/FastMCP 库自身
# 可能写 stderr)。保留这个保护层。
# ──────────────────────────────────────────────────────────────
try:
    _SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_SERVER_DIR, "data")
    os.makedirs(_DATA_DIR, exist_ok=True)
    _STDERR_LOG = os.path.join(_DATA_DIR, "stderr.log")
    _fd = os.open(_STDERR_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(_fd, 2)
    os.close(_fd)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# HF offline + sys.path cleanup (thin client 也可能间接 import 这些库)
# CUDA preload 不需要 (thin client 不加载模型)
# ──────────────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import asyncio
import logging

# Load .env and setup paths
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env_loader import setup, resolve_config  # noqa: E402
setup()

# Remove polluting Python 3.14 paths
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
os.environ.pop("AKARI_MEM_LIBS", None)

from mcp.server.fastmcp import FastMCP  # noqa: E402
from daemon_client import get_client, DaemonError  # noqa: E402

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")


def load_config() -> dict:
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


config = load_config()

logging.basicConfig(
    level=getattr(logging, config.get("log_level", "INFO")),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("akari-mem-mcp")

# Daemon client singleton (lazy ensure_running on first tool call)
_client = get_client(config)


def _ensure_daemon() -> bool:
    """Lazy ensure daemon running. Called at startup AND from each tool as safety net."""
    return _client.ensure_running(auto_spawn=True)


# ──────────────────────────────────────────────────────────────
# MCP Server
# ──────────────────────────────────────────────────────────────
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


def _err_text(prefix: str, e: Exception) -> str:
    return f"{prefix}: {e}"


@mcp.tool()
async def quick_search(
    query: str, limit: int = 5, project: str = "", tags: str = ""
) -> str:
    """
    Fast keyword search (FTS5). Millisecond-level, no model loading.
    Best for: exact name matching, specific terms, simple lookups.
    Use this FIRST. Escalate to search_memory if results are poor.

    Args:
        query: Search keywords (space-separated, OR logic)
        limit: Max results (default 5)
        project: Filter by project name (optional)
        tags: Filter by tag keyword (optional)
    """
    def _impl():
        try:
            rows = _client.quick_search(query, limit=limit, project=project, tags=tags)
        except DaemonError as e:
            return _err_text("[daemon error]", e)
        if not rows:
            return "No keyword matches. Try search_memory for semantic search."
        lines = [f"[quick] Found {len(rows)} matches:\n"]
        for r in rows:
            preview = (r.get("text") or "")[:100].replace("\n", " ")
            tags_str = f"  Tags: {r['tags']}" if r.get("tags") else ""
            lines.append(f"#{r['id']} | {r['title']} | {preview}...{tags_str}")
        return "\n".join(lines)

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def search_memory(
    query: str, limit: int = 5, project: str = "", tags: str = ""
) -> str:
    """
    Deep hybrid search: vector(BGE-M3) + keyword(FTS5) + RRF fusion + rerank.
    Takes 5-15s on first call (model loading), ~1-3s after warm-up.
    Best for: semantic/conceptual queries, fuzzy matching, complex questions.

    Args:
        query: Natural language search query
        limit: Max results (default 5)
        project: Filter by project name (optional)
        tags: Filter by tag keyword (optional)
    """
    def _impl():
        try:
            rows = _client.deep_search(query, limit=limit, project=project, tags=tags)
        except DaemonError as e:
            return _err_text("[daemon error]", e)
        if not rows:
            return "No memories found."
        lines = [f"[deep] Found {len(rows)} memories (use get_memory(id) for full content):\n"]
        for r in rows:
            rrf = r.get("rrf_score", "-")
            preview = (r.get("text") or "")[:100].replace("\n", " ")
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
        try:
            row = _client.get_memory(memory_id)
        except DaemonError as e:
            return _err_text("[daemon error]", e)
        if row is None:
            return f"Memory #{memory_id} not found."
        lines = [
            f"=== Memory #{row['id']}: {row['title']} ===",
            row.get("text", ""),
            "",
        ]
        if row.get("tags"):
            lines.append(f"Tags: {row['tags']}")
        if row.get("project"):
            lines.append(f"Project: {row['project']}")
        if row.get("source"):
            lines.append(f"Source: {row['source']}")
        if row.get("created_at"):
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
    def _impl():
        try:
            result = _client.save_memory(
                title=title, text=text, tags=tags, project=project, source="mcp",
            )
        except DaemonError as e:
            return _err_text("[daemon error]", e)
        mid = result.get("id")
        pending = result.get("pending_index", 0)
        warmup = result.get("warmup", "?")
        status = f"Memory saved: #{mid} — {title}"
        if warmup != "ready":
            status += f" (vector indexing queued, models {warmup})"
        elif pending > 1:
            status += f" ({pending} pending index)"
        return status

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def list_memories(limit: int = 10) -> str:
    """
    List the most recent memories.

    Args:
        limit: Number of memories to show (default 10)
    """
    def _impl():
        try:
            rows = _client.list_memories(limit)
        except DaemonError as e:
            return _err_text("[daemon error]", e)
        if not rows:
            return "No memories stored yet."
        lines = [f"Latest {len(rows)} memories:\n"]
        for r in rows:
            tags_str = f" [{r['tags']}]" if r.get("tags") else ""
            created = (r.get("created_at") or "")[:10]
            lines.append(f"  #{r['id']}: {r['title']}{tags_str} ({created})")
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
        try:
            result = _client.delete_memory(memory_id)
        except DaemonError as e:
            return _err_text("[daemon error]", e)
        if result.get("deleted"):
            return f"Memory #{memory_id} deleted."
        return f"Memory #{memory_id} not found."

    return await asyncio.to_thread(_impl)


@mcp.tool()
async def get_stats() -> str:
    """Get memory store statistics: total count, embedding model, projects, etc."""
    def _impl():
        try:
            s = _client.get_stats()
        except DaemonError as e:
            return _err_text("[daemon error]", e)
        lines = [
            "=== Akari Memory Stats ===",
            f"  Total memories: {s.get('total', 0)}",
            f"  Embedding: {s.get('embedding', 'default')}",
            f"  Rerank: {s.get('rerank', 'none')}",
            f"  Latest: {s.get('latest', 'none')}",
            f"  Warmup: {s.get('warmup', '?')}"
            + (f" (error: {s['warmup_error']})" if s.get("warmup_error") else ""),
            f"  Index queue: {s.get('index_queue_size', 0)} pending",
            f"  Daemon pid: {s.get('pid', '?')}, uptime: {s.get('uptime_seconds', 0)}s",
        ]
        projects = s.get("projects") or {}
        if projects:
            lines.append("  Projects:")
            for proj, cnt in projects.items():
                lines.append(f"    {proj}: {cnt}")
        return "\n".join(lines)

    return await asyncio.to_thread(_impl)


# ──────────────────────────────────────────────────────────────
# CLI commands (本地一次性操作, 不走 daemon)
# ──────────────────────────────────────────────────────────────
def _local_store():
    """临时加载本地 store (仅用于 CLI 子命令)。"""
    # 这里允许 import 重型库, 因为 CLI 子命令是一次性的
    from embeddings import create_provider
    from rerank import create_reranker
    from store import MemoryStore
    p = create_provider(config.get("embedding", {}))
    r = create_reranker(config.get("rerank", {}))
    return MemoryStore(data_dir=config["data_dir"], embedding_provider=p, reranker=r)


def run_migrate():
    """Import from claude-mem. 本地一次性, 不走 daemon。"""
    from migrate import migrate_from_claude_mem
    migrate_from_claude_mem(_local_store())


def run_rebuild():
    _local_store().rebuild_vectors()


def run_rebuild_fts():
    _local_store().rebuild_fts()


def run_test():
    """Quick search test via daemon."""
    if not _ensure_daemon():
        print("Daemon not running, falling back to local store.")
        s = _local_store()
        queries = ["阿卡林的猫叫什么名字", "赋驾公司的老板", "TPM 职业转型", "RTX 5070 调优"]
        for q in queries:
            results = s.search(q, 3)
            print(f'Query: "{q}"')
            for r in results:
                print(f"  #{r['id']} [{r.get('distance')}] {r['title'][:50]}")
            print()
        return

    queries = ["阿卡林的猫叫什么名字", "赋驾公司的老板", "TPM 职业转型", "RTX 5070 调优"]
    stats = _client.get_stats()
    print(f"Daemon stats: {json.dumps(stats, ensure_ascii=False, indent=2)}\n")
    for q in queries:
        results = _client.deep_search(q, 3)
        print(f'Query: "{q}"')
        for r in results:
            print(f"  #{r['id']} [{r.get('distance')}] {r['title'][:50]}")
        print()


def main():
    if "--migrate" in sys.argv:
        run_migrate()
        return
    if "--rebuild" in sys.argv:
        run_rebuild()
        return
    if "--rebuild-fts" in sys.argv:
        run_rebuild_fts()
        return
    if "--test" in sys.argv:
        run_test()
        return

    # ── Default: MCP stdio thin client ──────────────────────
    # 启动时立即尝试 spawn/连接 daemon (失败也继续, 用户首次 tool call 会再试)
    try:
        if _ensure_daemon():
            logger.info(f"Daemon ready at {_client.base_url}")
        else:
            logger.warning(
                f"Daemon unreachable at {_client.base_url}. "
                "Tools will retry on first call."
            )
    except Exception as e:
        logger.warning(f"Daemon ensure failed at startup: {e}. Will retry per-tool.")

    # Start MCP stdio server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
