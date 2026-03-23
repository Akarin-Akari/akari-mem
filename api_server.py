# -*- coding: utf-8 -*-
"""
akari-mem HTTP API server.

Lightweight Flask-free HTTP server for Antigravity skill integration.
Only uses stdlib (http.server) + sqlite3 — NO model loading at startup.
Deep search routes trigger lazy model loading on first call.

Usage:
  python api_server.py              # Start on port 37800
  python api_server.py --port 9900  # Custom port

Endpoints:
  GET  /health              → {"status":"ok","total":24}
  GET  /list?limit=10       → [memories...]
  GET  /search?q=xxx&limit=5 → [quick FTS5 results...]
  GET  /deep?q=xxx&limit=5  → [hybrid+rerank results...] (slow first call)
  POST /save                → {"id":25} (json body: title, text, tags, project)
  DELETE /delete?id=5       → {"deleted":true}
  GET  /stats               → {stats object}
"""
import os, sys, json, sqlite3, logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_EXTRA_LIB = os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs")
if os.path.isdir(_EXTRA_LIB) and _EXTRA_LIB not in sys.path:
    sys.path.append(_EXTRA_LIB)

os.environ.setdefault("HF_HOME", r"F:\models")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")

def load_config():
    defaults = {"data_dir": os.path.join(_PROJECT_ROOT, "data"), "embedding": {"mode": "default"}}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            defaults.update(json.load(f))
    return defaults

config = load_config()
DB_PATH = os.path.join(config["data_dir"], "akari-mem.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [akari-api] %(message)s", handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger("akari-api")

# ── Lazy store (for deep search + save) ─────────────────────
_store = None
def get_store():
    global _store
    if _store is None:
        logger.info("Loading models for deep search (first use)...")
        from embeddings import create_provider
        from rerank import create_reranker
        from store import MemoryStore
        p = create_provider(config.get("embedding", {}))
        r = create_reranker(config.get("rerank", {}))
        _store = MemoryStore(data_dir=config["data_dir"], embedding_provider=p, reranker=r)
        logger.info("MemoryStore ready.")
    return _store

# ── Direct SQLite helpers (instant, no models) ──────────────
def _db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

def _list_recent(limit=10):
    db = _db()
    rows = db.execute("SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

def _quick_search(query, limit=5):
    db = _db()
    try:
        rows = db.execute(
            "SELECT m.* FROM memories m JOIN memories_fts f ON m.id = f.rowid "
            "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?", (query, limit)
        ).fetchall()
    except Exception:
        terms = [t.strip() for t in query.split() if t.strip()]
        fts_q = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = db.execute(
                "SELECT m.* FROM memories m JOIN memories_fts f ON m.id = f.rowid "
                "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?", (fts_q, limit)
            ).fetchall()
        except Exception:
            rows = []
    db.close()
    return [dict(r) for r in rows]

def _get_stats():
    db = _db()
    total = db.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()["cnt"]
    latest = db.execute("SELECT title FROM memories ORDER BY id DESC LIMIT 1").fetchone()
    projects = {}
    for row in db.execute("SELECT project, COUNT(*) as cnt FROM memories WHERE project != '' GROUP BY project"):
        projects[row["project"]] = row["cnt"]
    db.close()
    return {
        "total": total,
        "latest": latest["title"] if latest else None,
        "embedding": config.get("embedding", {}).get("model", "default"),
        "rerank": config.get("rerank", {}).get("model", "none"),
        "models_loaded": _store is not None,
        "projects": projects,
    }

def _delete(memory_id):
    db = _db()
    cur = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    ok = cur.rowcount > 0
    db.close()
    return ok


# ── HTTP Handler ────────────────────────────────────────────
class AkariMemHandler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path.rstrip("/")

        try:
            if path == "/health":
                stats = _get_stats()
                self._json({"status": "ok", "total": stats["total"]})

            elif path == "/list":
                limit = int(qs.get("limit", [10])[0])
                self._json(_list_recent(limit))

            elif path == "/search":
                q = qs.get("q", [""])[0]
                limit = int(qs.get("limit", [5])[0])
                if not q:
                    self._json({"error": "missing ?q= parameter"}, 400)
                    return
                self._json(_quick_search(q, limit))

            elif path == "/deep":
                q = qs.get("q", [""])[0]
                limit = int(qs.get("limit", [5])[0])
                if not q:
                    self._json({"error": "missing ?q= parameter"}, 400)
                    return
                results = get_store().search(q, limit)
                self._json(results)

            elif path == "/stats":
                self._json(_get_stats())

            else:
                self._json({"error": "not found", "endpoints": ["/health", "/list", "/search", "/deep", "/stats"]}, 404)
        except Exception as e:
            logger.error(f"GET {path}: {e}")
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            if path == "/save":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                title = body.get("title", "")
                text = body.get("text", "")
                tags = body.get("tags", "")
                project = body.get("project", "")
                if not title or not text:
                    self._json({"error": "title and text required"}, 400)
                    return
                mem_id = get_store().save(title, text, tags=tags, project=project, source="api")
                self._json({"id": mem_id, "title": title})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            logger.error(f"POST {path}: {e}")
            self._json({"error": str(e)}, 500)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path.rstrip("/")
        try:
            if path == "/delete":
                mid = int(qs.get("id", [0])[0])
                if not mid:
                    self._json({"error": "missing ?id= parameter"}, 400)
                    return
                ok = _delete(mid)
                self._json({"deleted": ok, "id": mid})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            logger.error(f"DELETE {path}: {e}")
            self._json({"error": str(e)}, 500)

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} {format % args}")


def main():
    port = 37800
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    server = HTTPServer(("127.0.0.1", port), AkariMemHandler)
    logger.info(f"akari-mem API server on http://127.0.0.1:{port}")
    logger.info(f"  /health  /list  /search?q=  /deep?q=  /stats")
    logger.info(f"  POST /save  DELETE /delete?id=")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
