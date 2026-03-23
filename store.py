"""
Dual-engine memory store: SQLite (structured) + ChromaDB (vector).

Write operations always sync both. Read operations use the appropriate engine:
- Semantic search → ChromaDB
- List/filter/stats → SQLite
"""
import sqlite3
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

logger = logging.getLogger("akari-mem.store")

# Default data directory (next to this file)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class MemoryStore:
    """Dual-engine memory store: SQLite + ChromaDB + optional Rerank."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, embedding_provider=None, reranker=None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.db_path = os.path.join(data_dir, "akari-mem.db")
        self.chroma_dir = os.path.join(data_dir, "chroma")

        # Embedding
        from embeddings import ChromaEmbeddingAdapter, DefaultEmbeddingProvider

        self._provider = embedding_provider or DefaultEmbeddingProvider()
        self._chroma_ef = ChromaEmbeddingAdapter(self._provider)

        # Reranker (optional)
        self._reranker = reranker

        # Init
        self._init_sqlite()
        self._init_chroma()
        rerank_info = f" | rerank={self._reranker.model_name}" if self._reranker else ""
        logger.info(
            f"MemoryStore ready: {self.db_path} | "
            f"embedding={self._provider.model_name} ({self._provider.dimension}d)"
            f"{rerank_info}"
        )

    # ── SQLite ──────────────────────────────────────────────

    def _init_sqlite(self):
        db = sqlite3.connect(self.db_path)
        db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                text        TEXT NOT NULL,
                tags        TEXT DEFAULT '',
                project     TEXT DEFAULT '',
                source      TEXT DEFAULT 'manual',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        # FTS5 index for keyword search
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(title, text, tags, content='memories', content_rowid='id')
        """)
        # Triggers to keep FTS in sync
        db.executescript("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, title, text, tags)
                VALUES (new.id, new.title, new.text, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, title, text, tags)
                VALUES ('delete', old.id, old.title, old.text, old.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, title, text, tags)
                VALUES ('delete', old.id, old.title, old.text, old.tags);
                INSERT INTO memories_fts(rowid, title, text, tags)
                VALUES (new.id, new.title, new.text, new.tags);
            END;
        """)
        # Metadata table for tracking embedding model
        db.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        db.commit()
        db.close()

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    # ── ChromaDB ────────────────────────────────────────────

    def _init_chroma(self):
        import chromadb

        self._chroma_client = chromadb.PersistentClient(path=self.chroma_dir)

        # Check if existing collection uses different model
        existing_model = self._get_meta("embedding_model")
        current_model = self._provider.model_name

        if existing_model and existing_model != current_model:
            logger.warning(
                f"Embedding model changed: {existing_model} → {current_model}. "
                f"Deleting old collection (dimension may have changed)."
            )
            try:
                self._chroma_client.delete_collection("akari_memories")
                logger.info("Old ChromaDB collection deleted.")
            except Exception:
                pass

        self._collection = self._chroma_client.get_or_create_collection(
            name="akari_memories",
            embedding_function=self._chroma_ef,
            metadata={"hnsw:space": "cosine"},
        )

        self._set_meta("embedding_model", current_model)

    # ── Meta helpers ────────────────────────────────────────

    def _get_meta(self, key: str) -> Optional[str]:
        db = self._db()
        row = db.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        db.close()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str):
        db = self._db()
        db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        db.commit()
        db.close()

    # ── Public API ──────────────────────────────────────────

    def save(
        self,
        title: str,
        text: str,
        tags: str = "",
        project: str = "",
        source: str = "manual",
    ) -> int:
        """Save a memory to both SQLite and ChromaDB. Returns the new ID."""
        now = datetime.now(timezone.utc).isoformat()
        db = self._db()
        cur = db.execute(
            "INSERT INTO memories (title, text, tags, project, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, text, tags, project, source, now, now),
        )
        mem_id = cur.lastrowid
        db.commit()
        db.close()

        # Sync to ChromaDB
        document = f"{title}\n{text}"
        self._collection.add(
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

        logger.info(f"Saved memory #{mem_id}: {title[:40]}")
        return mem_id

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search: vector + keyword + RRF fusion + optional rerank.

        Stage 1a: ChromaDB vector search (semantic recall)
        Stage 1b: FTS5 keyword search (exact keyword recall)
        Stage 2:  RRF fusion (merge & deduplicate)
        Stage 3:  Reranker re-scores (if enabled)
        """
        # How many candidates to fetch from each source
        fetch_k = limit * 3 if self._reranker else limit * 2
        fetch_k = max(fetch_k, 10)  # at least 10 candidates for good fusion

        # ── Stage 1a: Vector recall ────────────────────────
        vector_results = []
        chroma_count = self._collection.count()
        if chroma_count > 0:
            query_vec = self._provider.embed([query])[0]
            vr = self._collection.query(
                query_embeddings=[query_vec],
                n_results=min(fetch_k, chroma_count),
            )
            if vr["ids"] and vr["ids"][0]:
                db = self._db()
                for i, cid in enumerate(vr["ids"][0]):
                    meta = vr["metadatas"][0][i]
                    sqlite_id = meta.get("sqlite_id")
                    distance = vr["distances"][0][i] if vr["distances"] else None
                    row = db.execute(
                        "SELECT * FROM memories WHERE id=?", (sqlite_id,)
                    ).fetchone()
                    if row:
                        vector_results.append({
                            "id": row["id"],
                            "title": row["title"],
                            "text": row["text"],
                            "tags": row["tags"],
                            "project": row["project"],
                            "source": row["source"],
                            "created_at": row["created_at"],
                            "distance": round(distance, 4) if distance else None,
                        })
                db.close()

        # ── Stage 1b: Keyword recall (FTS5) ────────────────
        keyword_results = self._keyword_search_safe(query, fetch_k)

        # ── Stage 2: RRF Fusion ────────────────────────────
        merged = self._rrf_fusion(vector_results, keyword_results, k=60)

        # ── Stage 3: Rerank (if enabled) ───────────────────
        if self._reranker and merged:
            merged = self._reranker.rerank(query, merged, top_k=limit)
            logger.debug(f"Reranked → {len(merged)} results")

        return merged[:limit]

    def _keyword_search_safe(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        FTS5 keyword search with graceful fallback.
        FTS5 MATCH syntax can fail on certain queries — fall back silently.
        """
        db = self._db()
        try:
            # Try exact FTS5 match first
            rows = db.execute(
                "SELECT m.* FROM memories m "
                "JOIN memories_fts f ON m.id = f.rowid "
                "WHERE memories_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        except Exception:
            # FTS5 syntax error — try splitting into OR terms
            try:
                terms = [t.strip() for t in query.split() if t.strip()]
                fts_query = " OR ".join(f'"{t}"' for t in terms)
                rows = db.execute(
                    "SELECT m.* FROM memories m "
                    "JOIN memories_fts f ON m.id = f.rowid "
                    "WHERE memories_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_query, limit),
                ).fetchall()
            except Exception:
                rows = []
        db.close()
        return [dict(r) for r in rows]

    @staticmethod
    def _rrf_fusion(
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion: merge two ranked lists.
        RRF score = sum(1 / (k + rank)) for each list the item appears in.
        Higher score = more relevant.
        """
        scores: Dict[int, float] = {}
        docs: Dict[int, Dict[str, Any]] = {}

        # Score vector results
        for rank, doc in enumerate(vector_results):
            mid = doc["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
            docs[mid] = doc

        # Score keyword results
        for rank, doc in enumerate(keyword_results):
            mid = doc["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
            if mid not in docs:
                doc["distance"] = None  # no vector distance for keyword-only hits
                docs[mid] = doc

        # Sort by RRF score descending
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result = []
        for mid in ranked_ids:
            d = docs[mid].copy()
            d["rrf_score"] = round(scores[mid], 6)
            result.append(d)

        return result

    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Public keyword search (for MCP tool or direct use)."""
        return self._keyword_search_safe(query, limit)

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List most recent memories."""
        db = self._db()
        rows = db.execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def get(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get a single memory by ID."""
        db = self._db()
        row = db.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        db.close()
        return dict(row) if row else None

    def delete(self, memory_id: int) -> bool:
        """Delete from both SQLite and ChromaDB."""
        db = self._db()
        cur = db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        db.commit()
        db.close()

        if cur.rowcount > 0:
            try:
                self._collection.delete(ids=[f"mem_{memory_id}"])
            except Exception:
                pass  # ChromaDB may not have it
            logger.info(f"Deleted memory #{memory_id}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        db = self._db()
        total = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        projects = db.execute(
            "SELECT project, COUNT(*) as cnt FROM memories "
            "GROUP BY project ORDER BY cnt DESC"
        ).fetchall()
        latest = db.execute(
            "SELECT created_at FROM memories ORDER BY id DESC LIMIT 1"
        ).fetchone()
        db.close()

        return {
            "total_memories": total,
            "chroma_count": self._collection.count(),
            "embedding_model": self._provider.model_name,
            "embedding_dim": self._provider.dimension,
            "rerank_model": self._reranker.model_name if self._reranker else "none",
            "projects": {r[0] or "(none)": r[1] for r in projects},
            "latest_memory": latest[0] if latest else None,
            "data_dir": self.data_dir,
        }

    def rebuild_vectors(self):
        """Re-embed all memories in ChromaDB. Use after changing embedding model."""
        logger.info("Rebuilding vector index...")

        # Delete all existing
        existing = self._collection.get()
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        # Re-add from SQLite
        db = self._db()
        rows = db.execute("SELECT * FROM memories ORDER BY id").fetchall()
        db.close()

        batch_size = 10
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._collection.add(
                ids=[f"mem_{r['id']}" for r in batch],
                documents=[f"{r['title']}\n{r['text']}" for r in batch],
                metadatas=[{
                    "sqlite_id": r["id"],
                    "title": r["title"][:200],
                    "tags": r["tags"] or "",
                    "project": r["project"] or "",
                    "source": r["source"] or "",
                } for r in batch],
            )
            logger.info(f"  Re-embedded batch {i // batch_size + 1}")

        self._set_meta("embedding_model", self._provider.model_name)
        logger.info(
            f"Rebuild complete: {len(rows)} memories re-embedded "
            f"with {self._provider.model_name}"
        )
