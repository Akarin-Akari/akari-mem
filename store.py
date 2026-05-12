"""
Dual-engine memory store: SQLite (structured) + ChromaDB (vector).

Write operations always sync both. Read operations use the appropriate engine:
- Semantic search → ChromaDB
- List/filter/stats → SQLite

FTS5 indexing uses jieba pre-tokenization for CJK support.
Triggers are removed; FTS5 is managed manually in Python.
"""
import sqlite3
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from tokenizer import tokenize_for_fts, tokenize_query

logger = logging.getLogger("akari-mem.store")

# Default data directory (next to this file)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _trace(msg: str) -> None:
    """Append a timestamped line to data/warmup.log so we can pinpoint hangs
    even when stdio is broken. Same file as the Step 1/2/3 markers."""
    try:
        import time as _t
        _log_path = os.path.join(DEFAULT_DATA_DIR, "warmup.log")
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{_t.strftime('%H:%M:%S')}] [store] {msg}\n")
            f.flush()
    except Exception:
        pass


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
        _trace("Step 3a: _init_sqlite start")
        self._init_sqlite()
        _trace("Step 3a: _init_sqlite done")
        _trace("Step 3b: _init_chroma start")
        self._init_chroma()
        _trace("Step 3b: _init_chroma done")
        rerank_info = f" | rerank={self._reranker.model_name}" if self._reranker else ""
        _trace("Step 3c: probing provider.dimension (may trigger model _load)")
        _dim = self._provider.dimension
        _trace(f"Step 3c: dimension={_dim} done")
        logger.info(
            f"MemoryStore ready: {self.db_path} | "
            f"embedding={self._provider.model_name} ({_dim}d)"
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
        # FTS5 index for keyword search (managed manually with jieba tokenization)
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(title, text, tags, content='memories', content_rowid='id')
        """)
        # Drop legacy auto-triggers (jieba tokenization requires Python-layer control)
        db.executescript("""
            DROP TRIGGER IF EXISTS memories_ai;
            DROP TRIGGER IF EXISTS memories_ad;
            DROP TRIGGER IF EXISTS memories_au;
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
        _trace("  3b.1: import chromadb")
        import chromadb

        _trace("  3b.2: PersistentClient() (may hang on telemetry/WAL)")
        self._chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        _trace("  3b.2: PersistentClient done")

        # Check if existing collection uses different model
        _trace("  3b.3: _get_meta(embedding_model)")
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

        _trace("  3b.4: get_or_create_collection (may probe embedding_function)")
        self._collection = self._chroma_client.get_or_create_collection(
            name="akari_memories",
            embedding_function=self._chroma_ef,
            metadata={"hnsw:space": "cosine"},
        )
        _trace("  3b.4: get_or_create_collection done")

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

        # Sync FTS5 index with jieba-tokenized text
        self._fts_insert(db, mem_id, title, text, tags)

        db.commit()
        db.close()

        # Sync to ChromaDB with chunking
        from chunker import chunk_text

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
            self._collection.add(
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
            self._collection.add(ids=ids, documents=chunks, metadatas=metas)

        logger.info(f"Saved memory #{mem_id}: {title[:40]} ({len(chunks)} chunk(s))")
        return mem_id

    def search(
        self,
        query: str,
        limit: int = 5,
        project: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: vector + FTS5 keyword, merged via RRF.
        If reranker is configured, re-scores the merged list.
        Supports optional project/tags pre-filtering.

        Returns list of memory dicts sorted by relevance.
        """
        # Determine how many candidates to fetch
        fetch_k = limit * 3 if self._reranker else limit * 2
        fetch_k = max(fetch_k, 10)  # at least 10 candidates for good fusion

        # Build ChromaDB where filter for metadata pre-filtering
        chroma_where = None
        where_clauses = []
        if project:
            where_clauses.append({"project": project})
        if tags:
            where_clauses.append({"tags": {"$contains": tags}})
        if len(where_clauses) == 1:
            chroma_where = where_clauses[0]
        elif len(where_clauses) > 1:
            chroma_where = {"$and": where_clauses}

        # ── Stage 1a: Vector recall ────────────────────────
        vector_results = []
        chroma_count = self._collection.count()
        if chroma_count > 0:
            query_vec = self._provider.embed([query])[0]
            query_kwargs = {
                "query_embeddings": [query_vec],
                "n_results": min(fetch_k * 2, chroma_count),
            }
            if chroma_where:
                query_kwargs["where"] = chroma_where
            try:
                vr = self._collection.query(**query_kwargs)
            except Exception:
                # Filter may fail if no matching docs — fallback to unfiltered
                vr = self._collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(fetch_k * 2, chroma_count),
                )
            if vr["ids"] and vr["ids"][0]:
                db = self._db()
                # Chunk dedup: keep best distance per sqlite_id
                seen_ids: Dict[int, float] = {}  # sqlite_id -> best_distance
                candidates = []
                for i, cid in enumerate(vr["ids"][0]):
                    meta = vr["metadatas"][0][i]
                    sqlite_id = meta.get("sqlite_id")
                    distance = vr["distances"][0][i] if vr["distances"] else None
                    # Dedup: only keep the chunk with smallest distance
                    if sqlite_id in seen_ids:
                        if distance is not None and distance < seen_ids[sqlite_id]:
                            seen_ids[sqlite_id] = distance
                            candidates = [c for c in candidates if c["id"] != sqlite_id]
                        else:
                            continue
                    else:
                        seen_ids[sqlite_id] = distance if distance is not None else float("inf")

                    row = db.execute(
                        "SELECT * FROM memories WHERE id=?", (sqlite_id,)
                    ).fetchone()
                    if row:
                        candidates.append({
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
                vector_results = candidates[:fetch_k]

        # ── Stage 1b: Keyword recall (FTS5) ────────────────
        keyword_results = self._keyword_search_safe(
            query, fetch_k, project=project, tags=tags
        )

        # ── Stage 2: RRF Fusion ────────────────────────────
        merged = self._rrf_fusion(vector_results, keyword_results, k=60)

        # ── Stage 3: Rerank (if enabled) ───────────────────
        if self._reranker and merged:
            merged = self._reranker.rerank(query, merged, top_k=limit)
            logger.debug(f"Reranked → {len(merged)} results")

        return merged[:limit]

    # ── FTS5 manual sync (jieba tokenization) ───────────────

    @staticmethod
    def _fts_insert(db: sqlite3.Connection, mem_id: int, title: str, text: str, tags: str):
        """Insert jieba-tokenized content into FTS5 index."""
        tok_title = tokenize_for_fts(title)
        tok_text = tokenize_for_fts(text)
        # tags are comma-separated ASCII-ish, no need for jieba
        db.execute(
            "INSERT INTO memories_fts(rowid, title, text, tags) VALUES (?, ?, ?, ?)",
            (mem_id, tok_title, tok_text, tags),
        )

    @staticmethod
    def _fts_delete(db: sqlite3.Connection, mem_id: int, title: str, text: str, tags: str):
        """Remove entry from FTS5 index (content-sync requires matching values)."""
        tok_title = tokenize_for_fts(title)
        tok_text = tokenize_for_fts(text)
        db.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, title, text, tags) "
            "VALUES ('delete', ?, ?, ?, ?)",
            (mem_id, tok_title, tok_text, tags),
        )

    def _keyword_search_safe(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        FTS5 keyword search with jieba tokenization and graceful fallback.
        Query is tokenized with jieba before MATCH for CJK compatibility.
        Supports optional project/tags pre-filtering.
        """
        db = self._db()

        # Tokenize query with jieba for CJK support
        tokenized_query = tokenize_query(query)

        # Build optional SQL WHERE clause for metadata filtering
        extra_where = ""
        extra_params: list = []
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

        try:
            rows = db.execute(base_sql, (tokenized_query, *extra_params, limit)).fetchall()
        except Exception:
            # FTS5 syntax error — try splitting into OR terms
            try:
                terms = [t.strip() for t in tokenized_query.split() if t.strip()]
                fts_query = " OR ".join(f'"{t}"' for t in terms)
                rows = db.execute(base_sql, (fts_query, *extra_params, limit)).fetchall()
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

    def keyword_search(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Public keyword search (for MCP tool or direct use)."""
        return self._keyword_search_safe(query, limit, project=project, tags=tags)

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
        """Delete from SQLite, FTS5, and ChromaDB (including all chunks)."""
        db = self._db()

        # Read original data first (needed for FTS5 content-sync delete)
        row = db.execute(
            "SELECT title, text, tags FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if not row:
            db.close()
            return False

        # Delete from FTS5 first (requires original values for content-sync)
        try:
            self._fts_delete(db, memory_id, row["title"], row["text"], row["tags"])
        except Exception as e:
            logger.warning(f"FTS5 delete for #{memory_id} failed (non-fatal): {e}")

        # Delete from SQLite
        db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        db.commit()
        db.close()

        # Delete from ChromaDB
        try:
            self._collection.delete(where={"sqlite_id": memory_id})
        except Exception:
            try:
                self._collection.delete(ids=[f"mem_{memory_id}"])
            except Exception:
                pass

        logger.info(f"Deleted memory #{memory_id} (including chunks)")
        return True

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

    def rebuild_fts(self):
        """Rebuild FTS5 index with jieba tokenization. Use after first installing jieba."""
        logger.info("Rebuilding FTS5 index with jieba tokenization...")
        db = self._db()

        # Purge existing FTS5 index
        db.execute("INSERT INTO memories_fts(memories_fts) VALUES ('delete-all')")

        # Re-index all memories with jieba tokenization
        rows = db.execute("SELECT id, title, text, tags FROM memories ORDER BY id").fetchall()
        for r in rows:
            self._fts_insert(db, r["id"], r["title"], r["text"], r["tags"])

        db.commit()
        db.close()
        logger.info(f"FTS5 rebuild complete: {len(rows)} memories re-indexed with jieba")

    def rebuild_vectors(self):
        """Re-embed all memories in ChromaDB with chunking. Use after changing embedding model."""
        from chunker import chunk_text

        logger.info("Rebuilding vector index (with chunking)...")

        # Delete all existing
        existing = self._collection.get()
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        # Re-add from SQLite
        db = self._db()
        rows = db.execute("SELECT * FROM memories ORDER BY id").fetchall()
        db.close()

        total_chunks = 0
        for r in rows:
            document = f"{r['title']}\n{r['text']}"
            chunks = chunk_text(document)
            base_meta = {
                "sqlite_id": r["id"],
                "title": r["title"][:200],
                "tags": r["tags"] or "",
                "project": r["project"] or "",
                "source": r["source"] or "",
            }

            if len(chunks) <= 1:
                self._collection.add(
                    ids=[f"mem_{r['id']}"],
                    documents=[document],
                    metadatas=[{**base_meta, "chunk_index": 0, "total_chunks": 1}],
                )
            else:
                ids = [f"mem_{r['id']}_chunk_{i}" for i in range(len(chunks))]
                metas = [
                    {**base_meta, "chunk_index": i, "total_chunks": len(chunks)}
                    for i in range(len(chunks))
                ]
                self._collection.add(ids=ids, documents=chunks, metadatas=metas)

            total_chunks += len(chunks)

        self._set_meta("embedding_model", self._provider.model_name)
        logger.info(
            f"Rebuild complete: {len(rows)} memories → {total_chunks} chunks "
            f"re-embedded with {self._provider.model_name}"
        )
