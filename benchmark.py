# -*- coding: utf-8 -*-
"""Final benchmark - minimal output to avoid terminal truncation."""
import sys, os
_extra = os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.append(_extra)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from store import MemoryStore
from embeddings import create_provider
from rerank import create_reranker

p = create_provider({"mode": "local", "model": "BAAI/bge-m3", "cache_dir": "F:/models"})
r = create_reranker({"mode": "local", "model": "BAAI/bge-reranker-v2-m3", "cache_dir": "F:/models"})
s = MemoryStore(data_dir="F:/claude-tools/akari-mem-mcp/data", embedding_provider=p, reranker=r)

cases = [
    ("\u963f\u5361\u6797\u7684\u732b\u53eb\u4ec0\u4e48\u540d\u5b57", "\u4e2a\u4eba\u6863\u6848"),
    ("\u8d4b\u9a7e\u516c\u53f8\u7684\u8001\u677f\u662f\u8c01", "\u804c\u4e1a\u5c65\u5386"),
    ("TPM \u804c\u4e1a\u8f6c\u578b \u8d74\u65e5", "\u804c\u4e1a\u65b9\u5411"),
    ("ACG \u6e38\u620f \u660e\u65e5\u65b9\u821f", "\u5174\u8da3\u7231\u597d"),
    ("\u6df1\u5733 \u623f\u4ea7 \u5bb6\u5ead", "\u5bb6\u5ead\u4e0e\u8d22\u52a1"),
    ("RTX 5070 \u7b14\u8bb0\u672c\u8c03\u4f18", "\u6280\u672f\u753b\u50cf"),
    ("\u963f\u5361\u6797\u8ba8\u538c\u4ec0\u4e48", "\u4ef7\u503c\u89c2"),
    ("Claude Code \u64a4\u9500\u5de5\u5177", "ccundo"),
    ("\u5de5\u5177\u4f18\u5148\u7ea7 ripgrep", "\u4ee3\u7801\u641c\u7d22\u5de5\u5177\u4f18\u5148\u7ea7"),
    ("\u65e5\u8bed\u6c34\u5e73 N2", "\u4e2a\u4eba\u6863\u6848"),
]

hits = 0
for q, exp in cases:
    res = s.search(q, 3)
    t = res[0] if res else None
    ok = t and exp in t["title"]
    if ok:
        hits += 1
    tag = "Y" if ok else "X"
    tid = t["id"] if t else 0
    tt = t["title"][:30] if t else "NONE"
    print(f"[{tag}] #{tid} {tt}")

print(f"\nSCORE: {hits}/10 = {hits*10}%")
