import sys, os, json
# Append (not insert) F:\python-libs so system-wide packages take priority
_extra = os.environ.get("AKARI_MEM_LIBS", r"F:\python-libs")
if os.path.isdir(_extra) and _extra not in sys.path:
    sys.path.append(_extra)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from store import MemoryStore
from embeddings import create_provider

config_path = os.path.join(os.path.dirname(__file__), "config.json")
config = {}
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

provider = create_provider(config.get("embedding", {}))
data_dir = config.get("data_dir", os.path.join(os.path.dirname(__file__), "data"))
store = MemoryStore(data_dir=data_dir, embedding_provider=provider)

# Stats
stats = store.get_stats()
print(f"=== Akari-Mem Stats ===")
print(f"  Total: {stats['total_memories']} memories")
print(f"  Chroma: {stats['chroma_count']} vectors")
print(f"  Model: {stats['embedding_model']} ({stats['embedding_dim']}d)")
print()

# Search tests
queries = [
    "阿卡林的猫叫什么名字",
    "赋驾公司的老板是谁",
    "TPM 职业转型 赴日",
    "ACG 游戏 明日方舟",
    "深圳 房产 家庭",
    "RTX 5070 笔记本调优",
]

for q in queries:
    results = store.search(q, 3)
    print(f'Query: "{q}"')
    for r in results:
        print(f"  #{r['id']} [dist={r['distance']}] {r['title'][:55]}")
    print()
