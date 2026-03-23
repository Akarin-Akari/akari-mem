# -*- coding: utf-8 -*-
"""Deep diagnosis: what does each memory contain? Why do queries miss?"""
import sys, os
_extra = os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.append(_extra)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from store import MemoryStore
from embeddings import create_provider

cfg = {"mode": "local", "model": "BAAI/bge-m3", "cache_dir": "F:/models"}
p = create_provider(cfg)
s = MemoryStore(data_dir="F:/claude-tools/akari-mem-mcp/data", embedding_provider=p)

# 1. List all memories with title + text preview
print("=== ALL MEMORIES ===")
all_mems = s.list_recent(100)
for m in sorted(all_mems, key=lambda x: x["id"]):
    text_preview = m["text"][:80].replace("\n", " ")
    print(f"  #{m['id']:2d} | {m['title'][:50]:50s} | {text_preview}...")
print()

# 2. Deep search for the 3 missed queries - show top 5 with distance
missed = [
    "ACG anime arknights",
    "shenzhen housing property family",
    "RTX 5070 laptop GPU tuning",
]
# Also Chinese versions
missed_cn = [
    "ACG 动画 明日方舟 游戏",
    "深圳 房产 家庭 财务",
    "RTX 5070 笔记本 GPU 调优",
]

for q in missed_cn:
    results = s.search(q, 5)
    print(f'Q: "{q}"')
    for r in results:
        print(f"  #{r['id']:2d} [d={r['distance']:.4f}] {r['title'][:55]}")
    print()
