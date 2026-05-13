# -*- coding: utf-8 -*-
"""
akari-mem HTTP daemon server.

ROLE (升级后职责):
=================
- 唯一加载 BGE-M3 + jina-reranker 模型的进程（GPU 共享）
- 同时为以下客户端提供 HTTP 服务:
    * MCP thin client (server.py) — Claude Code agents
    * Antigravity skill — 本地 GET/POST 调用
- Windows DETACHED_PROCESS 静默后台运行
- 单例机制: socket bind 抢占 (端口被占用即退出)

启动方式:
  python api_server.py                 # 前台启动 (调试)
  python api_server.py --daemon        # 后台 spawn 时的标志位
  python api_server.py --port 9900     # 自定义端口

Endpoints (向后兼容 + MCP 新增):
  GET    /health                                      → 健康检查（含 warmup 状态）
  GET    /list?limit=N                                → 最近 N 条
  GET    /search?q=&limit=&project=&tags=             → 快速 FTS5 搜索 (jieba 分词)
  GET    /deep?q=&limit=&project=&tags=               → 深度向量+keyword+rerank 搜索
  GET    /memory?id=N                                 → 单条完整内容 (MCP 新增)
  POST   /save  body:{title,text,tags,project}        → 异步保存 (SQLite 即时 + ChromaDB 排队)
  DELETE /delete?id=N                                 → 同步删除 (SQLite + ChromaDB)
  GET    /stats                                       → 统计信息
  POST   /shutdown                                    → 优雅停机 (MCP 新增)
"""
import os
import sys

# ──────────────────────────────────────────────────────────────
# CRITICAL: Redirect OS-level stderr (fd 2) to a file BEFORE any imports
# 与 server.py 的逻辑保持一致：native libraries (ORT, cuDNN, TRT, transformers)
# 写 fd 2 不走 Python，如果上游不消费会阻塞数十分钟。
# 注意：daemon 进程 stderr 也要重定向，避免 DETACHED_PROCESS 下的 fd 异常。
# ──────────────────────────────────────────────────────────────
try:
    _SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_SERVER_DIR, "data")
    os.makedirs(_DATA_DIR, exist_ok=True)
    _STDERR_LOG = os.path.join(_DATA_DIR, "daemon-stderr.log")
    _fd = os.open(_STDERR_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(_fd, 2)
    os.close(_fd)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace")
except Exception:
    pass

# stdout 同样重定向 (DETACHED_PROCESS 下 stdout 也无效)
try:
    _STDOUT_LOG = os.path.join(_DATA_DIR, "daemon-stdout.log")
    _fd_out = os.open(_STDOUT_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(_fd_out, 1)
    os.close(_fd_out)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# CRITICAL: Preload CUDA + cuDNN DLLs from PyTorch's bundled runtime
# 完整从 server.py 搬过来 — 这是 GPU EP 工作的关键。
# 详细 rationale 见 server.py 顶部 / CLAUDE.md。
# ──────────────────────────────────────────────────────────────
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    try:
        import ctypes as _ctypes
        _venv_site = os.path.join(
            os.path.dirname(sys.executable), "..", "Lib", "site-packages"
        )
        _torch_lib = os.path.normpath(os.path.join(_venv_site, "torch", "lib"))
        if os.path.isdir(_torch_lib):
            try:
                os.add_dll_directory(_torch_lib)
            except Exception:
                pass
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
                        pass
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# Silence model loading output BEFORE any imports
# ──────────────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ.get("AKARI_MODEL_CACHE", ""))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sqlite3
import logging
import socket
import threading
import queue as _queue_mod
import time
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env_loader import setup, resolve_config
setup()

# ──────────────────────────────────────────────────────────────
# CRITICAL: Remove polluting global Python 3.14 paths from sys.path
# 与 server.py 保持一致
# ──────────────────────────────────────────────────────────────
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

# Pre-import heavy libs that write to stderr at import time
try:
    import chromadb  # noqa: F401 — grpc/protobuf init writes to stderr
except ImportError:
    pass

from tokenizer import tokenize_for_fts, tokenize_query

CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")

def load_config():
    defaults = {
        "data_dir": os.path.join(_PROJECT_ROOT, "data"),
        "embedding": {"mode": "default"},
        "log_level": "INFO",
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            defaults.update(json.load(f))
    return resolve_config(defaults)


config = load_config()
DATA_DIR = config["data_dir"]
DB_PATH = os.path.join(DATA_DIR, "akari-mem.db")
PIDFILE = os.path.join(DATA_DIR, "daemon.pid")

# Daemon config (host/port/idle_shutdown)
_daemon_cfg = config.get("daemon", {}) if isinstance(config, dict) else {}
DAEMON_HOST = _daemon_cfg.get("host", "127.0.0.1")
DAEMON_PORT = int(_daemon_cfg.get("port", 37800))
DAEMON_IDLE_SHUTDOWN = float(_daemon_cfg.get("idle_shutdown_seconds", 1800.0))

# Pre-import torch only when needed (local embedding/rerank mode)
def _peek_mode(section: str) -> str:
    try:
        return str(config.get(section, {}).get("mode", "")).lower()
    except Exception:
        return ""

_emb_mode = _peek_mode("embedding")
_rnk_mode = _peek_mode("rerank")
_needs_torch = _emb_mode == "local" or _rnk_mode == "local"

if _needs_torch:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

logging.basicConfig(
    level=getattr(logging, config.get("log_level", "INFO")),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("akari-daemon")


# ──────────────────────────────────────────────────────────────
# Store + Background Workers (从 server.py 搬过来)
# ──────────────────────────────────────────────────────────────
_store = None
_store_lock = threading.Lock()
_warmup_state = "idle"          # idle → loading → ready | failed
_warmup_error: str = ""
_index_queue: "_queue_mod.Queue" = _queue_mod.Queue()
_last_request_at = 0.0          # 用于 daemon 自我关闭
_request_lock = threading.Lock()

# Concurrency control for deep search (one ONNX session at a time)
_deep_search_lock = threading.Lock()


def _touch_request():
    """每个 HTTP 请求都会 touch，用于 idle-shutdown / idle-unload。"""
    global _last_request_at
    with _request_lock:
        _last_request_at = time.monotonic()


# ──────────────────────────────────────────────────────────────
# Idle-Unload Watchdog (从 server.py 搬过来)
# ──────────────────────────────────────────────────────────────
class IdleUnloadManager:
    def __init__(self, idle_timeout_seconds=600.0, check_interval=60.0):
        self.idle_timeout = idle_timeout_seconds
        self.check_interval = check_interval
        self._last_activity = 0.0
        self._lock = threading.Lock()
        self._targets = []
        self._stop = threading.Event()
        self._thread = None
        self.enabled = True

    def register(self, name, get_provider_fn):
        self._targets.append((name, get_provider_fn))

    def touch(self):
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
        while not self._stop.wait(self.check_interval):
            if not self.enabled:
                continue
            with self._lock:
                last = self._last_activity
            if last == 0.0:
                continue
            idle_for = time.monotonic() - last
            if idle_for < self.idle_timeout:
                continue
            for name, get_provider in self._targets:
                try:
                    p = get_provider()
                    if p is None:
                        continue
                    if hasattr(p, "is_loaded") and p.is_loaded():
                        logger.info(
                            f"[idle-unload] {name} idle for {idle_for:.0f}s, unloading"
                        )
                        p.unload()
                except Exception as e:
                    logger.warning(f"[idle-unload] failed to unload {name}: {e}")
            with self._lock:
                self._last_activity = 0.0


_iu_cfg = config.get("idle_unload", {}) if isinstance(config, dict) else {}
_iu_enabled = bool(_iu_cfg.get("enabled", True))
_iu_timeout = float(_iu_cfg.get("timeout_seconds", 600.0))
_iu_targets = list(_iu_cfg.get("targets", ["reranker", "embedding"]))

_unload_mgr = IdleUnloadManager(
    idle_timeout_seconds=_iu_timeout,
    check_interval=min(60.0, _iu_timeout / 10.0),
)
_unload_mgr.enabled = _iu_enabled


def _register_unload_targets():
    if "embedding" in _iu_targets:
        _unload_mgr.register(
            "embedding",
            lambda: (_store._provider if _store is not None else None),
        )
    if "reranker" in _iu_targets:
        _unload_mgr.register(
            "reranker",
            lambda: (_store._reranker if _store is not None else None),
        )


# ──────────────────────────────────────────────────────────────
# Store accessor + warmup (从 server.py 搬过来, 精简)
# ──────────────────────────────────────────────────────────────
def _warmup_log(msg):
    try:
        with open(os.path.join(DATA_DIR, "warmup.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            f.flush()
    except Exception:
        pass


def get_store(timeout=60.0):
    """获取 MemoryStore 全局单例，与 warmup 线程协作。"""
    global _store
    if _store is not None:
        return _store

    # warmup 在跑就等
    if _warmup_state == "loading":
        deadline = time.time() + timeout
        while _store is None and _warmup_state == "loading":
            if time.time() > deadline:
                logger.warning(f"get_store: warmup did not finish within {timeout}s")
                break
            time.sleep(0.05)
        if _store is not None:
            return _store

    # fallback: 同步构建
    with _store_lock:
        if _store is None:
            from embeddings import create_provider
            from rerank import create_reranker
            from store import MemoryStore
            p = create_provider(config.get("embedding", {}))
            r = create_reranker(config.get("rerank", {}))
            _store = MemoryStore(data_dir=DATA_DIR, embedding_provider=p, reranker=r)
    return _store


def _background_warmup():
    global _warmup_state, _warmup_error, _store
    time.sleep(0.5)  # let HTTP server start
    _warmup_state = "loading"
    try:
        _warmup_log("Warmup started")
        from embeddings import create_provider
        from rerank import create_reranker
        from store import MemoryStore as _MS
        _warmup_log("Step 1: create_provider...")
        p = create_provider(config.get("embedding", {}))
        _warmup_log(f"Step 1 done: {type(p).__name__}")
        _warmup_log("Step 2: create_reranker...")
        r = create_reranker(config.get("rerank", {}))
        _warmup_log(f"Step 2 done: {type(r).__name__}")
        _warmup_log("Step 3: MemoryStore init...")
        _store_obj = _MS(data_dir=DATA_DIR, embedding_provider=p, reranker=r)
        _warmup_log("Step 3 done")
        _store = _store_obj
        _warmup_state = "ready"
        _warmup_log("Warmup COMPLETE")
    except Exception as e:
        _warmup_state = "failed"
        _warmup_error = str(e)
        import traceback
        _warmup_log(f"Warmup FAILED: {e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────
# Save fast-path: SQLite immediate + ChromaDB async queue
# (从 server.py 搬过来)
# ──────────────────────────────────────────────────────────────
def _save_to_sqlite(title, text, tags, project, source):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    db = sqlite3.connect(DB_PATH)
    cur = db.execute(
        "INSERT INTO memories (title, text, tags, project, source, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, text, tags, project, source, now, now),
    )
    mem_id = cur.lastrowid
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
    """后台线程: 消费 _index_queue 把 memory 写入 ChromaDB。"""
    from chunker import chunk_text
    while True:
        item = _index_queue.get()
        if item is None:
            break
        mem_id, title, text, tags, project, source = item
        try:
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
                store._collection.add(
                    ids=[f"mem_{mem_id}"],
                    documents=[document],
                    metadatas=[{**base_meta, "chunk_index": 0, "total_chunks": 1}],
                )
            else:
                ids = [f"mem_{mem_id}_chunk_{i}" for i in range(len(chunks))]
                metas = [
                    {**base_meta, "chunk_index": i, "total_chunks": len(chunks)}
                    for i in range(len(chunks))
                ]
                store._collection.add(ids=ids, documents=chunks, metadatas=metas)
            logger.info(f"Indexed memory #{mem_id} ({len(chunks)} chunks)")
        except Exception as e:
            logger.exception(f"Failed to index #{mem_id}: {e}")
        _index_queue.task_done()


# ──────────────────────────────────────────────────────────────
# Direct SQLite helpers (instant, no models)
# ──────────────────────────────────────────────────────────────
def _db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def _list_recent(limit=10):
    if not os.path.exists(DB_PATH):
        return []
    db = _db()
    rows = db.execute(
        "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def _quick_search(query, limit=5, project=None, tags=None):
    """FTS5 keyword search with jieba tokenization and optional filters."""
    if not os.path.exists(DB_PATH):
        return []
    tokenized = tokenize_query(query)

    extra_where = ""
    extra_params = []
    if project:
        extra_where += " AND m.project = ?"
        extra_params.append(project)
    if tags:
        extra_where += " AND m.tags LIKE ?"
        extra_params.append(f"%{tags}%")

    base_sql = (
        "SELECT m.* FROM memories m "
        "JOIN memories_fts f ON m.id = f.rowid "
        "WHERE memories_fts MATCH ?" + extra_where + " "
        "ORDER BY rank LIMIT ?"
    )
    db = _db()
    try:
        rows = db.execute(base_sql, (tokenized, *extra_params, limit)).fetchall()
    except Exception:
        # FTS5 syntax fallback → OR terms
        terms = [t.strip() for t in tokenized.split() if t.strip()]
        fts_q = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = db.execute(base_sql, (fts_q, *extra_params, limit)).fetchall()
        except Exception:
            rows = []
    db.close()
    return [dict(r) for r in rows]


def _get_memory(memory_id):
    if not os.path.exists(DB_PATH):
        return None
    db = _db()
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    db.close()
    return dict(row) if row else None


def _delete_full(memory_id):
    """同步删除 SQLite + ChromaDB（如果 store 已加载）。"""
    db = _db()
    cur = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    deleted = cur.rowcount > 0
    db.close()
    if deleted and _store is not None:
        try:
            _store._collection.delete(where={"sqlite_id": memory_id})
        except Exception:
            pass
    return deleted


def _get_stats():
    if not os.path.exists(DB_PATH):
        return {"total": 0, "warmup": _warmup_state, "models_loaded": False}
    db = _db()
    total = db.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()["cnt"]
    latest = db.execute("SELECT title FROM memories ORDER BY id DESC LIMIT 1").fetchone()
    projects = {}
    for row in db.execute(
        "SELECT project, COUNT(*) as cnt FROM memories WHERE project != '' GROUP BY project"
    ):
        projects[row["project"]] = row["cnt"]
    db.close()
    return {
        "total": total,
        "latest": latest["title"] if latest else None,
        "embedding": config.get("embedding", {}).get("model", "default"),
        "rerank": config.get("rerank", {}).get("model", "none"),
        "models_loaded": _store is not None,
        "warmup": _warmup_state,
        "warmup_error": _warmup_error or None,
        "index_queue_size": _index_queue.qsize(),
        "projects": projects,
        "pid": os.getpid(),
        "uptime_seconds": int(time.monotonic() - _START_TIME),
    }


# ──────────────────────────────────────────────────────────────
# HTTP Handler
# ──────────────────────────────────────────────────────────────
class AkariMemHandler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _touch_request()
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path == "/health":
                stats = _get_stats()
                self._json({
                    "status": "ok",
                    "total": stats["total"],
                    "warmup": stats["warmup"],
                    "models_loaded": stats["models_loaded"],
                })

            elif path == "/list":
                limit = int(qs.get("limit", [10])[0])
                self._json(_list_recent(limit))

            elif path == "/search":
                q = qs.get("q", [""])[0]
                limit = int(qs.get("limit", [5])[0])
                project = qs.get("project", [""])[0] or None
                tags = qs.get("tags", [""])[0] or None
                if not q:
                    self._json({"error": "missing ?q= parameter"}, 400)
                    return
                self._json(_quick_search(q, limit, project=project, tags=tags))

            elif path == "/deep":
                q = qs.get("q", [""])[0]
                limit = int(qs.get("limit", [5])[0])
                project = qs.get("project", [""])[0] or None
                tags = qs.get("tags", [""])[0] or None
                if not q:
                    self._json({"error": "missing ?q= parameter"}, 400)
                    return
                try:
                    _unload_mgr.touch()
                except Exception:
                    pass
                # 串行化深度搜索 (一次只一个 ONNX 推理, 防爆 VRAM)
                with _deep_search_lock:
                    results = get_store().search(q, limit, project=project, tags=tags)
                self._json(results)

            elif path == "/memory":
                try:
                    mid = int(qs.get("id", ["0"])[0])
                except ValueError:
                    self._json({"error": "invalid id"}, 400)
                    return
                if not mid:
                    self._json({"error": "missing ?id= parameter"}, 400)
                    return
                rec = _get_memory(mid)
                if rec is None:
                    self._json({"error": "not found", "id": mid}, 404)
                else:
                    self._json(rec)

            elif path == "/stats":
                self._json(_get_stats())

            else:
                self._json({
                    "error": "not found",
                    "endpoints": [
                        "/health", "/list", "/search", "/deep",
                        "/memory", "/stats",
                    ],
                }, 404)
        except Exception as e:
            logger.exception(f"GET {path}: {e}")
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        _touch_request()
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/save":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                title = body.get("title", "")
                text = body.get("text", "")
                tags = body.get("tags", "")
                project = body.get("project", "")
                source = body.get("source", "api")
                if not title or not text:
                    self._json({"error": "title and text required"}, 400)
                    return
                # Fast path: SQLite + FTS5 同步落盘
                mem_id = _save_to_sqlite(title, text, tags, project, source)
                # 异步排队 ChromaDB 索引
                try:
                    _unload_mgr.touch()
                except Exception:
                    pass
                _index_queue.put((mem_id, title, text, tags, project, source))
                self._json({
                    "id": mem_id,
                    "title": title,
                    "queued": True,
                    "pending_index": _index_queue.qsize(),
                    "warmup": _warmup_state,
                })

            elif path == "/shutdown":
                logger.info("Shutdown requested via HTTP")
                self._json({"status": "shutting down", "pid": os.getpid()})
                # 延迟关闭 (让响应发出去)
                threading.Thread(
                    target=_graceful_shutdown, daemon=True, name="shutdown"
                ).start()

            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            logger.exception(f"POST {path}: {e}")
            self._json({"error": str(e)}, 500)

    def do_DELETE(self):
        _touch_request()
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/delete":
                try:
                    mid = int(qs.get("id", ["0"])[0])
                except ValueError:
                    self._json({"error": "invalid id"}, 400)
                    return
                if not mid:
                    self._json({"error": "missing ?id= parameter"}, 400)
                    return
                ok = _delete_full(mid)
                self._json({"deleted": ok, "id": mid})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            logger.exception(f"DELETE {path}: {e}")
            self._json({"error": str(e)}, 500)

    def log_message(self, fmt, *args):
        # 避免给 stderr 写太多 (DETACHED_PROCESS 下 stderr 重定向到文件)
        try:
            logger.info(f"{self.client_address[0]} {fmt % args}")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# Daemon lifecycle
# ──────────────────────────────────────────────────────────────
_START_TIME = time.monotonic()
_httpd: "HTTPServer | None" = None


def _self_shutdown_watchdog():
    """如果 idle_shutdown_seconds 内无请求，daemon 自我退出释放资源。"""
    if DAEMON_IDLE_SHUTDOWN <= 0:
        return
    # 给 daemon 至少 60 秒 grace period (期间没请求也别立刻退)
    time.sleep(60.0)
    check_interval = min(60.0, DAEMON_IDLE_SHUTDOWN / 10.0)
    while True:
        time.sleep(check_interval)
        with _request_lock:
            last = _last_request_at
        idle = time.monotonic() - last if last > 0 else 0
        # 首次启动后还没收到任何请求 → 也按 idle 计时 (last=0 时用 _START_TIME)
        if last == 0:
            idle = time.monotonic() - _START_TIME
        if idle >= DAEMON_IDLE_SHUTDOWN:
            logger.info(
                f"Daemon idle for {idle:.0f}s "
                f"(>= {DAEMON_IDLE_SHUTDOWN:.0f}s), self-shutting-down"
            )
            _graceful_shutdown()
            return


def _graceful_shutdown():
    """优雅停机: shutdown HTTP server → 清理 pidfile → exit。"""
    global _httpd
    try:
        # 停 index worker
        try:
            _index_queue.put(None)
        except Exception:
            pass
        # 停 idle-unload watchdog
        try:
            _unload_mgr.stop()
        except Exception:
            pass
        if _httpd is not None:
            _httpd.shutdown()
    except Exception:
        pass
    # 删 pidfile
    try:
        if os.path.exists(PIDFILE):
            os.unlink(PIDFILE)
    except Exception:
        pass
    # 给主线程一点时间 cleanup
    time.sleep(0.5)
    os._exit(0)


def _check_port_available(host, port) -> bool:
    """快速检查端口是否被占用。返回 True 表示可用。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Windows 上 SO_EXCLUSIVEADDRUSE 防止 SO_REUSEADDR 滥用
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def _write_pidfile(port):
    try:
        with open(PIDFILE, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n{port}\n{int(time.time())}\n")
    except Exception:
        pass


def main():
    global _httpd

    port = DAEMON_PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    # 单例检查: 端口被占用 = 已有 daemon 在跑, 静默退出
    if not _check_port_available(DAEMON_HOST, port):
        logger.info(
            f"Port {DAEMON_HOST}:{port} already in use — "
            f"another daemon instance is running. Exiting."
        )
        sys.exit(0)

    try:
        _httpd = HTTPServer((DAEMON_HOST, port), AkariMemHandler)
    except OSError as e:
        # 二次校验 (race: 别人在两次 check 之间抢占了 port)
        logger.info(f"Failed to bind {DAEMON_HOST}:{port}: {e}. Another daemon won the race.")
        sys.exit(0)

    _write_pidfile(port)
    logger.info(f"akari-mem daemon started on http://{DAEMON_HOST}:{port} (pid={os.getpid()})")
    logger.info(
        f"  config: embedding.mode={_emb_mode}, rerank.mode={_rnk_mode}, "
        f"idle_unload={_iu_enabled} ({_iu_timeout:.0f}s), "
        f"idle_shutdown={DAEMON_IDLE_SHUTDOWN:.0f}s"
    )

    # 启动后台线程
    threading.Thread(target=_background_warmup, daemon=True, name="warmup").start()
    threading.Thread(target=_index_worker, daemon=True, name="indexer").start()
    if _iu_enabled:
        _register_unload_targets()
        _unload_mgr.start()
        logger.info(f"Idle-unload watchdog started: timeout={_iu_timeout:.0f}s")
    if DAEMON_IDLE_SHUTDOWN > 0:
        threading.Thread(
            target=_self_shutdown_watchdog,
            daemon=True,
            name="self-shutdown",
        ).start()
        logger.info(f"Self-shutdown watchdog: idle_timeout={DAEMON_IDLE_SHUTDOWN:.0f}s")

    # 注册 SIGTERM/SIGINT 处理
    def _on_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down")
        threading.Thread(target=_graceful_shutdown, daemon=True).start()
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGINT, _on_signal)
    except Exception:
        pass

    try:
        _httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down")
    finally:
        _graceful_shutdown()


if __name__ == "__main__":
    main()
