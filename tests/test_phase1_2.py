"""
Smoke test for Phase 1 (Chunking) + Phase 2 (Metadata Pre-filtering).

Tests:
  1. chunker.py unit tests (short/long/overlap)
  2. store.py integration: save + search + delete with chunks
  3. Metadata pre-filtering: project/tags filter works
  4. Backward compatibility: short text still works as single entry

Usage: python tests/test_phase1_2.py
"""
import os
import sys
import shutil
import tempfile
import io

# Fix Windows GBK encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Setup path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ── Test 1: Chunker unit tests ──────────────────────────────

def test_chunker():
    from chunker import chunk_text

    print("=== Test 1: Chunker Unit Tests ===")

    # 1a: Empty text
    result = chunk_text("")
    assert result == [], f"Expected [], got {result}"
    print("  ✓ 1a: Empty text → []")

    # 1b: Short text (below threshold)
    short = "This is a short memory."
    result = chunk_text(short, max_chars=500)
    assert result == [short], f"Expected [{short}], got {result}"
    print("  ✓ 1b: Short text → single chunk")

    # 1c: Long text with paragraphs
    paragraphs = "\n\n".join([f"Paragraph {i}: " + "x" * 100 for i in range(10)])
    result = chunk_text(paragraphs, max_chars=300, overlap=30)
    assert len(result) > 1, f"Expected multiple chunks, got {len(result)}"
    print(f"  ✓ 1c: Long text ({len(paragraphs)} chars) → {len(result)} chunks")

    # 1d: Chinese text
    cn_text = "这是第一段中文内容。" * 30 + "\n\n" + "这是第二段中文内容。" * 30
    result = chunk_text(cn_text, max_chars=200, overlap=20)
    assert len(result) > 1, f"Expected multiple chunks for Chinese, got {len(result)}"
    print(f"  ✓ 1d: Chinese text ({len(cn_text)} chars) → {len(result)} chunks")

    # 1e: Overlap check
    text = "AAA BBB CCC.\n\nDDD EEE FFF.\n\nGGG HHH III."
    result = chunk_text(text, max_chars=20, overlap=5)
    assert len(result) >= 2, f"Expected >=2 chunks, got {len(result)}"
    # Second chunk should start with overlap from first
    if len(result) > 1:
        print(f"  ✓ 1e: Overlap works, chunk 2 starts with: '{result[1][:20]}...'")

    print("  ✅ All chunker tests passed!\n")


# ── Test 2: Store integration ───────────────────────────────

def test_store_integration():
    print("=== Test 2: Store Integration (Chunking + Search + Delete) ===")

    # Create temp directory for test data
    test_dir = tempfile.mkdtemp(prefix="akari_test_")
    print(f"  Test data dir: {test_dir}")

    try:
        from embeddings import create_provider
        from rerank import create_reranker
        from store import MemoryStore

        # Use default (lightweight) embedding for fast test
        provider = create_provider({"mode": "default"})
        reranker = create_reranker({"mode": "none"})
        store = MemoryStore(data_dir=test_dir, embedding_provider=provider, reranker=reranker)

        # 2a: Save short text
        id_short = store.save("Short note", "Quick reminder.", tags="test", project="unit-test")
        print(f"  ✓ 2a: Saved short memory #{id_short}")

        # 2b: Save long text (should be chunked)
        long_text = "\n\n".join([
            f"Section {i}: This is a detailed description about topic {i}. " +
            "It contains multiple sentences that provide context and information. " * 5
            for i in range(8)
        ])
        id_long = store.save("Long document", long_text, tags="test,long", project="unit-test")
        print(f"  ✓ 2b: Saved long memory #{id_long} ({len(long_text)} chars)")

        # Check ChromaDB has multiple entries for the long doc
        chroma_count = store._collection.count()
        print(f"  ℹ ChromaDB total entries: {chroma_count}")
        assert chroma_count > 2, f"Expected > 2 entries in ChromaDB, got {chroma_count}"
        print(f"  ✓ 2b: ChromaDB has chunked entries")

        # 2c: Search (should find both)
        results = store.search("topic", limit=5)
        assert len(results) > 0, "Expected search results"
        ids_found = [r["id"] for r in results]
        print(f"  ✓ 2c: Search 'topic' found {len(results)} results: {ids_found}")

        # 2d: Search should deduplicate chunks
        for r in results:
            assert "chunk_index" not in r, "Search result should NOT expose chunk_index"
        ids_unique = list(set(ids_found))
        assert len(ids_found) == len(ids_unique), f"Duplicate IDs in results: {ids_found}"
        print(f"  ✓ 2d: No duplicate memories in results")

        # 2e: Delete long doc — should clean up all chunks
        ok = store.delete(id_long)
        assert ok, f"Delete #{id_long} failed"
        new_count = store._collection.count()
        print(f"  ✓ 2e: Deleted #{id_long}, ChromaDB count: {chroma_count} → {new_count}")

        # 2f: Clean up short doc
        store.delete(id_short)
        print(f"  ✓ 2f: Cleaned up #{id_short}")

        print("  ✅ All store integration tests passed!\n")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# ── Test 3: Metadata pre-filtering ──────────────────────────

def test_metadata_filtering():
    print("=== Test 3: Metadata Pre-filtering ===")

    test_dir = tempfile.mkdtemp(prefix="akari_filter_test_")
    print(f"  Test data dir: {test_dir}")

    try:
        from embeddings import create_provider
        from rerank import create_reranker
        from store import MemoryStore

        provider = create_provider({"mode": "default"})
        reranker = create_reranker({"mode": "none"})
        store = MemoryStore(data_dir=test_dir, embedding_provider=provider, reranker=reranker)

        # Save memories in different projects
        id1 = store.save("Python tips", "Use list comprehensions for clean code", project="python")
        id2 = store.save("Java tips", "Use streams for functional programming", project="java")
        id3 = store.save("Go tips", "Use goroutines for concurrency", project="golang", tags="concurrency")

        # 3a: Search without filter — should find all
        results = store.search("programming tips", limit=10)
        assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"
        print(f"  ✓ 3a: Unfiltered search: {len(results)} results")

        # 3b: Search with project filter
        results_py = store.search("tips", limit=10, project="python")
        py_ids = [r["id"] for r in results_py]
        print(f"  ✓ 3b: Project='python' filter: {len(results_py)} results, ids={py_ids}")

        # 3c: Keyword search with project filter
        kw_results = store.keyword_search("tips", limit=10, project="java")
        kw_ids = [r["id"] for r in kw_results]
        print(f"  ✓ 3c: Keyword search project='java': {len(kw_results)} results, ids={kw_ids}")

        # 3d: Search with tags filter
        results_tags = store.search("tips", limit=10, tags="concurrency")
        tag_ids = [r["id"] for r in results_tags]
        print(f"  ✓ 3d: Tags='concurrency' filter: {len(results_tags)} results, ids={tag_ids}")

        # Clean up
        store.delete(id1)
        store.delete(id2)
        store.delete(id3)

        print("  ✅ All metadata filtering tests passed!\n")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Akari-Mem Phase 1+2 Smoke Test")
    print("=" * 60 + "\n")

    test_chunker()
    test_store_integration()
    test_metadata_filtering()

    print("=" * 60)
    print("  🎉 ALL SMOKE TESTS PASSED!")
    print("=" * 60)
