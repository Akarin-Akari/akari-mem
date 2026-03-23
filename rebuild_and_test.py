"""Rebuild BGE-M3 vectors and run quality test."""
import sys, os
# System packages first, F:\python-libs as fallback
_extra = os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.append(_extra)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from store import MemoryStore
from embeddings import create_provider

cfg = {"mode": "local", "model": "BAAI/bge-m3", "cache_dir": "F:/models"}
p = create_provider(cfg)
s = MemoryStore(data_dir="F:/claude-tools/akari-mem-mcp/data", embedding_provider=p)

# Rebuild
print("Rebuilding vectors with BGE-M3...")
s.rebuild_vectors()

# Stats
stats = s.get_stats()
print(f"\nTotal: {stats['total_memories']} | Chroma: {stats['chroma_count']}")
print(f"Model: {stats['embedding_model']} ({stats['embedding_dim']}d)")

# Search tests
queries = [
    "cats pets",
    "job company boss",
    "career japan",
    "anime game arknights",
    "shenzhen housing family",
    "RTX laptop tuning",
]
print()
for q in queries:
    results = s.search(q, 3)
    print(f'Q: "{q}"')
    for r in results:
        dist = r.get("distance", "?")
        print(f"  #{r['id']} [d={dist}] {r['title'][:55]}")
    print()
