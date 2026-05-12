"""
akari-mem-mcp — MCP memory server entry point.

Usage:
  python server.py                    # Start stdio MCP server
  python server.py --migrate          # Import from claude-mem first, then start
  python server.py --rebuild          # Rebuild vector index, then start
  python server.py --test             # Run quick search test
"""
import os
import sys

# ── CRITICAL: Redirect OS-level stderr (fd 2) to a file BEFORE any imports ──
# Background: When running under MCP stdio transport, sys.stderr is a pipe
# to the host (Claude Code). Native libraries (ORT, cuDNN, TRT, transformers)
# write warnings directly to OS fd 2. On Windows, the pipe buffer is ~64KB.
# If the host doesn't consume fast enough, the write blocks — freezing the
# warmup thread mid-load (observed: Step 3 hangs for 10-17 minutes).
# Redirecting fd 2 to a regular file at startup eliminates this entirely
# while still letting Python logging (which we redirect below) work.
# Done before any other import so even import-time native warnings are safe.
try:
    _SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_SERVER_DIR, "data")
    os.makedirs(_DATA_DIR, exist_ok=True)
    _STDERR_LOG = os.path.join(_DATA_DIR, "stderr.log")
    _fd = os.open(_STDERR_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(_fd, 2)
    os.close(_fd)
    # Reattach Python's sys.stderr to the (now-file-backed) fd 2 so Python
    # writes also go to the file rather than the original pipe.
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace")
except Exception:
    # If redirect fails, fall back to the original pipe and pray.
    pass

# ── CRITICAL: Preload CUDA + cuDNN DLLs from PyTorch's bundled runtime ──
# Why this exists:
#   onnxruntime-gpu 1.23.x ships `onnxruntime_providers_cuda.dll` which
#   depends on a TWO-MAJOR-VERSION mix of CUDA libs: cublas/cudart/cusparse
#   use CUDA 12 (cublasLt64_12.dll, cudart64_12.dll), but cufft/cusolver use
#   CUDA 11 (cufft64_11.dll, cusolver64_11.dll), plus cuDNN 9.x.
#
#   The PyTorch GPU wheel (`torch-X.Y.Z+cuXXX`) ships ALL of these under
#   `<venv>/Lib/site-packages/torch/lib/` — a complete CUDA 12 + 11 + cuDNN 9
#   runtime (~2 GB). Because it's vendored inside the venv, the user does
#   NOT need to install system CUDA Toolkit or cuDNN. Everything just works
#   as long as PyTorch's DLLs get loaded into the process BEFORE ORT tries
#   to load its provider DLL.
#
# Why `os.add_dll_directory` is not enough:
#   `add_dll_directory` only affects LoadLibrary calls that opt into
#   `LOAD_LIBRARY_SEARCH_USER_DIRS`. When ORT's native loader pulls in
#   `onnxruntime_providers_cuda.dll`, that DLL's own dependency resolution
#   (cublasLt, cufft, cudnn, ...) follows Windows' default DLL search order,
#   which does NOT consult user dirs. Result: silent fallback to CPUExecutionProvider
#   (RAM blows up, VRAM stays at zero, embedding latency ~50s).
#
# Why explicit ctypes preload works:
#   `ctypes.WinDLL(absolute_path)` loads the DLL into the process and pins
#   it in memory under its base name. When ORT later asks Windows to resolve
#   the same name as a transitive dependency, the loader sees it's already
#   present and reuses the loaded module — no PATH search needed.
#
# Background — earlier behavior:
#   Before we switched the reranker to fastembed-CPU, `import torch` happened
#   first as part of the sentence-transformers reranker load path. PyTorch's
#   own initialization preloads all these DLLs internally, so ORT inherited
#   them "for free". After the reranker change, torch is never imported on
#   the ORT load path, and the GPU silently dropped out.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    try:
        import ctypes as _ctypes
        _venv_site = os.path.join(
            os.path.dirname(sys.executable), "..", "Lib", "site-packages"
        )
        _torch_lib = os.path.normpath(
            os.path.join(_venv_site, "torch", "lib")
        )
        if os.path.isdir(_torch_lib):
            # Register the directory so any subsequent LoadLibrary that DOES
            # honor user dirs (Python's own ctypes / extension loaders) can
            # find these DLLs by base name too.
            try:
                os.add_dll_directory(_torch_lib)
            except Exception:
                pass
            # Preload order: low-level → high-level. Anything that depends on
            # something else must come after its dependency.
            _PRELOAD_DLLS = (
                "cudart64_12.dll",
                "nvrtc64_120_0.dll",
                "nvrtc-builtins64_128.dll",
                "nvJitLink_120_0.dll",
                "cublas64_12.dll",
                "cublasLt64_12.dll",
                "cufft64_11.dll",
                "cufftw64_11.dll",
                "curand64_10.dll",
                "cusparse64_12.dll",
                "cusolver64_11.dll",
                "cusolverMg64_11.dll",
                "cudnn64_9.dll",
                "cudnn_graph64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_heuristic64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll",
                "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll",
            )
            for _dll in _PRELOAD_DLLS:
                _path = os.path.join(_torch_lib, _dll)
                if os.path.exists(_path):
                    try:
                        _ctypes.WinDLL(_path)
                    except OSError:
                        # Missing or version-skewed DLL — ORT will report a
                        # clearer error later when it tries to use it.
                        pass
    except Exception:
        # Any failure here = ORT will fall back to CPU EP, which still works
        # (just slower). We don't want to crash server startup over this.
        pass

# ── CRITICAL: Silence model loading output BEFORE any imports ────
# MCP uses stdio pipes. Model loading (HF, tqdm, transformers) writes
# progress to stdout/stderr. If the pipe buffer fills and the host
# doesn't consume fast enough, the warmup thread blocks forever.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ.get("AKARI_MODEL_CACHE", ""))
# Force pure-offline mode for HuggingFace libraries. Even with local model
# files, `from_pretrained()` will (by default) make a HEAD request to verify
# freshness. Slow/blocked huggingface.co or hf-mirror.com hangs warmup for
# minutes. These two vars short-circuit ALL network probes.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
from tokenizer import tokenize_for_fts, tokenize_query

# Pre-import heavy libs at module level (before MCP stdio takes over).
# These imports trigger grpc/protobuf/tqdm/CUDA init that writes to
# stdout/stderr. If done inside the warmup thread (after MCP starts),
# the output fills the pipe buffer and blocks the thread forever.
#
# IMPORTANT: Only pre-import what the configured backends actually need.
# Pre-importing torch under ONNX+FastEmbed mode wastes ~0.5-1 GB of VRAM
# on PyTorch's CUDA context (cuBLAS/cuDNN lazy load), even though no
# PyTorch model is ever loaded.
try:
    import chromadb  # noqa: F401 — grpc/protobuf init writes to stderr
except ImportError:
    pass

def _peek_mode(section: str) -> str:
    """Read mode from config.json without triggering full config load."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
                _cfg = json.load(_f)
            return str(_cfg.get(section, {}).get("mode", "")).lower()
    except Exception:
        pass
    return ""

# CONFIG_PATH is defined below; resolve it inline here to avoid forward ref
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")

_emb_mode = _peek_mode("embedding")
_rnk_mode = _peek_mode("rerank")
_needs_torch = _emb_mode == "local" or _rnk_mode == "local"

if _needs_torch:
    try:
        import sentence_transformers  # noqa: F401 — tqdm/transformers logging
    except ImportError:
        pass
    try:
        import torch  # noqa: F401 — CUDA init output
    except ImportError:
        pass
else:
    logger_msg = (
        f"Skipped torch/sentence_transformers preload "
        f"(embedding.mode={_emb_mode or '<unset>'}, rerank.mode={_rnk_mode or '<unset>'})"
    )
    # logger not yet configured, defer to first MCP startup log
    sys.stderr.write(f"[akari-mem] {logger_msg}\n")

# ── Config ──────────────────────────────────────────────────

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

# Concurrency control for deep search (vector + rerank).
# Created lazily on first MCP request so it binds to the running event loop.
# Why a semaphore here (not a threading.Lock inside embed/rerank)?
# - Lock inside model libs caused deadlocks last time (FastEmbed has its own
#   thread pool, re-entrant LSP-style calls into Lock fail).
# - Semaphore(1) at the MCP tool layer guarantees only ONE deep search
#   touches the ONNX session at a time, with zero coupling to model internals.
_deep_search_sem = None  # type: ignore[assignment]

def _get_deep_search_sem():
    """Lazy-create the semaphore so it binds to the active asyncio loop."""
    global _deep_search_sem
    if _deep_search_sem is None:
        _deep_search_sem = asyncio.Semaphore(1)
    return _deep_search_sem


# ── Idle-Unload Watchdog ─────────────────────────────────────
# Frees model memory after a period of inactivity. Designed to:
# 1. NOT take any lock that user-facing code might also hold.
# 2. Use swap-style unload (model providers null their refs); inflight
#    requests keep working via local references, GC reclaims later.
# 3. Re-load on next request is transparent (providers are lazy-loaded).

class IdleUnloadManager:
    def __init__(self, idle_timeout_seconds: float = 600.0, check_interval: float = 60.0):
        self.idle_timeout = idle_timeout_seconds
        self.check_interval = check_interval
        self._last_activity = 0.0  # 0 means "no activity yet, do not unload"
        self._lock = threading.Lock()  # protects _last_activity only — no model code holds this
        self._targets = []  # list of (name, get_provider_fn)
        self._stop = threading.Event()
        self._thread = None
        self.enabled = True

    def register(self, name: str, get_provider_fn):
        """Register a model provider. `get_provider_fn()` returns the provider
        object (or None if store not yet built). Provider must expose
        `is_loaded()` and `unload()` (swap-style)."""
        self._targets.append((name, get_provider_fn))

    def touch(self):
        """Mark recent activity. Called from MCP search/save handlers."""
        import time
        with self._lock:
            self._last_activity = time.monotonic()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="idle-unload-watchdog"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        import time
        # Wait for at least one activity before starting the idle clock —
        # avoids unloading the freshly warmed-up model that nobody has used yet.
        while not self._stop.wait(self.check_interval):
            if not self.enabled:
                continue
            with self._lock:
                last = self._last_activity
            if last == 0.0:
                continue  # no activity recorded yet
            idle_for = time.monotonic() - last
            if idle_for < self.idle_timeout:
                continue
            # Unload each registered target if loaded.
            for name, get_provider in self._targets:
                try:
                    p = get_provider()
                    if p is None:
                        continue
                    if hasattr(p, "is_loaded") and p.is_loaded():
                        logger.info(
                            f"[idle-unload] {name} idle for {idle_for:.0f}s "
                            f"(>{self.idle_timeout:.0f}s), unloading"
                        )
                        p.unload()
                except Exception as e:
                    logger.warning(f"[idle-unload] failed to unload {name}: {e}")
            # Reset last_activity so we don't spam unload attempts.
            with self._lock:
                self._last_activity = 0.0


# Read idle-unload config (with sensible defaults)
def _idle_unload_config() -> dict:
    cfg = config.get("idle_unload", {}) if isinstance(config, dict) else {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "timeout_seconds": float(cfg.get("timeout_seconds", 600.0)),
        "targets": list(cfg.get("targets", ["reranker", "embedding"])),
    }

_iu_cfg = _idle_unload_config()
_unload_mgr = IdleUnloadManager(
    idle_timeout_seconds=_iu_cfg["timeout_seconds"],
    check_interval=min(60.0, _iu_cfg["timeout_seconds"] / 10.0),
)
_unload_mgr.enabled = _iu_cfg["enabled"]

def _register_unload_targets():
    """Register embedding/reranker providers based on config."""
    targets = _iu_cfg["targets"]
    if "embedding" in targets:
        _unload_mgr.register(
            "embedding",
            lambda: (_store._provider if _store is not None else None),
        )
    if "reranker" in targets:
        _unload_mgr.register(
            "reranker",
            lambda: (_store._reranker if _store is not None else None),
        )

def get_store(timeout: float = 60.0):
    """
    Get the global MemoryStore. Race-safe vs background warmup:

    1. Fast path: store already built → return immediately
    2. If warmup is in progress → wait for it (up to `timeout` s) to avoid
       constructing a second MemoryStore (which would re-load ONNX session,
       double VRAM usage, and trigger SQLite WAL contention).
    3. Fallback: warmup failed or never started → construct under _store_lock.
    """
    global _store
    if _store is not None:
        return _store

    # If warmup is loading, wait for it. This is the critical fix:
    # without it, the first incoming MCP request races warmup and both
    # try to construct a MemoryStore concurrently.
    if _warmup_state == "loading":
        import time as _t
        deadline = _t.time() + timeout
        while _store is None and _warmup_state == "loading":
            if _t.time() > deadline:
                logger.warning(
                    f"get_store: warmup did not finish within {timeout}s, "
                    f"falling back to synchronous construction"
                )
                break
            _t.sleep(0.05)
        if _store is not None:
            return _store

    # Fallback: warmup never ran, failed, or timed out.
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
    """Fast SQLite-only save with FTS5 sync. Returns mem_id."""
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
    # Sync FTS5 index with jieba-tokenized text
    tok_title = tokenize_for_fts(title)
    tok_text = tokenize_for_fts(text)
    db.execute(
        "INSERT INTO memories_fts(rowid, title, text, tags) VALUES (?, ?, ?, ?)",
        (mem_id, tok_title, tok_text, tags),
    )
    db.commit()
    db.close()
    return mem_id

def _index_worker():
    """Background thread: picks up saved memories and indexes them in ChromaDB."""
    from chunker import chunk_text

    while True:
        item = _index_queue.get()
        if item is None:
            break  # shutdown signal
        mem_id, title, text, tags, project, source = item
        try:
            # Reset idle-unload timer so the embedding model stays hot
            # while there are pending index jobs.
            try:
                _unload_mgr.touch()
            except Exception:
                pass
            store = get_store()
            document = f"{title}\n{text}"
            chunks = chunk_text(document)
            base_meta = {
                "sqlite_id": mem_id,
                "title": title[:200],
                "tags": tags,
                "project": project,
                "source": source,
            }

            if len(chunks) <= 1:
                # Short text — single entry (backward compatible)
                store._collection.add(
                    ids=[f"mem_{mem_id}"],
                    documents=[document],
                    metadatas=[{**base_meta, "chunk_index": 0, "total_chunks": 1}],
                )
            else:
                # Long text — multiple chunks
                ids = [f"mem_{mem_id}_chunk_{i}" for i in range(len(chunks))]
                metas = [
                    {**base_meta, "chunk_index": i, "total_chunks": len(chunks)}
                    for i in range(len(chunks))
                ]
                store._collection.add(ids=ids, documents=chunks, metadatas=metas)

            logger.info(f"Indexed memory #{mem_id} in ChromaDB ({len(chunks)} chunk(s)).")
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
    _project = project or None
    _tags = tags or None

    def _impl():
        import sqlite3
        db_path = os.path.join(config["data_dir"], "akari-mem.db")
        if not os.path.exists(db_path):
            return "No memory database found."

        # Build optional WHERE clause for metadata filtering
        extra_where = ""
        extra_params: list = []
        if _project:
            extra_where += " AND m.project = ?"
            extra_params.append(_project)
        if _tags:
            extra_where += " AND m.tags LIKE ?"
            extra_params.append(f"%{_tags}%")

        base_sql = (
            "SELECT m.* FROM memories m "
            "JOIN memories_fts f ON m.id = f.rowid "
            "WHERE memories_fts MATCH ?" + extra_where + " "
            "ORDER BY rank LIMIT ?"
        )

        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row

        # Tokenize query with jieba for CJK support
        tokenized = tokenize_query(query)

        try:
            rows = db.execute(base_sql, (tokenized, *extra_params, limit)).fetchall()
        except Exception:
            # FTS5 syntax error — split into OR terms
            terms = [t.strip() for t in tokenized.split() if t.strip()]
            fts_q = " OR ".join(f'"{t}"' for t in terms)
            try:
                rows = db.execute(base_sql, (fts_q, *extra_params, limit)).fetchall()
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
    _project = project or None
    _tags = tags or None

    def _impl():
        # touch idle-unload watchdog (so frequent searches keep models hot)
        try:
            _unload_mgr.touch()  # noqa: F821 — defined below
        except Exception:
            pass
        results = get_store().search(query, limit, project=_project, tags=_tags)
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

    # Serialize deep searches at the MCP layer to prevent concurrent ONNX
    # inference / rerank from doubling VRAM. This is the safe replacement
    # for the old per-provider threading.Lock that caused deadlocks.
    async with _get_deep_search_sem():
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
    # save_memory enqueues background indexing which DOES touch the embedding
    # model, so reset the idle-unload timer to keep models hot.
    try:
        _unload_mgr.touch()
    except Exception:
        pass
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
    if "--rebuild-fts" in sys.argv:
        get_store().rebuild_fts()
    if "--test" in sys.argv:
        run_test()
        return

    # Start background workers
    threading.Thread(target=_background_warmup, daemon=True, name="warmup").start()
    threading.Thread(target=_index_worker, daemon=True, name="indexer").start()

    # Start idle-unload watchdog (frees model VRAM/RAM after inactivity)
    if _iu_cfg["enabled"]:
        _register_unload_targets()
        _unload_mgr.start()
        logger.info(
            f"Idle-unload watchdog started: timeout={_iu_cfg['timeout_seconds']:.0f}s, "
            f"targets={_iu_cfg['targets']}"
        )

    # Start MCP stdio server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
