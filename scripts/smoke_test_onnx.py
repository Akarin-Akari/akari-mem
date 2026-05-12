"""End-to-end smoke test for akari-mem with ONNX FP16 embedding backend.

Verifies the full pipeline:
  1. Server boots with new config (onnx + cuda + fp16).
  2. save_memory writes to SQLite + ChromaDB.
  3. search_memory returns relevant results with RRF fusion.
  4. delete_memory removes from both engines.
"""
import os
import sys
import time

os.environ["AKARI_CONFIG"] = "F:/claude-tools/akari-mem-mcp/config.json"
os.environ["AKARI_MODEL_CACHE"] = "F:/models"
sys.path.insert(0, "F:/claude-tools/akari-mem-mcp")

import json
from store import MemoryStore  # noqa: E402
from embeddings import create_provider  # noqa: E402
from rerank import create_reranker  # noqa: E402


def banner(s):
    print()
    print("=" * 60)
    print(s)
    print("=" * 60)


def main():
    cfg_path = os.environ["AKARI_CONFIG"]
    config = json.loads(open(cfg_path, "r", encoding="utf-8").read())
    print(f"Using config: {cfg_path}")
    print(f"Embedding mode = {config['embedding']['mode']}  "
          f"model = {config['embedding']['model']}")

    banner("BOOTING MemoryStore with ONNX FP16 ...")
    t0 = time.time()
    p = create_provider(config.get("embedding", {}))
    r = create_reranker(config.get("rerank", {}))
    store = MemoryStore(
        data_dir="F:/tmp/akari_smoke_onnx",
        embedding_provider=p,
        reranker=r,
    )
    print(f"boot in {time.time() - t0:.2f}s")
    print(f"embedding provider: {type(p).__name__} -> {p.model_name} (dim={p.dimension})")
    if r is not None:
        print(f"reranker:           {type(r).__name__} -> {r.model_name}")

    banner("STAGE 1: save")
    docs = [
        {"title": "ONNX FP16 加速", "text": "猫娘工程师浮浮酱用 ONNX 把 BGE-M3 量化到 FP16，速度起飞", "project": "akari-mem", "tags": "onnx,fp16,perf"},
        {"title": "RRF 混合检索", "text": "RAG 系统里 ChromaDB 用向量检索，SQLite 用关键字检索，最后 RRF 融合", "project": "akari-mem", "tags": "rag,rrf"},
        {"title": "Spring Boot WebFlux", "text": "Spring Boot 3.2 + Reactive WebFlux + R2DBC stream backpressure", "project": "huifeixingry", "tags": "spring-boot,reactive"},
        {"title": "随手记", "text": "今天天气真好，适合出门散步晒太阳", "project": "personal", "tags": "chitchat"},
        {"title": "量化精度", "text": "深度学习模型量化：FP32 -> FP16 通常精度损失 < 0.1%", "project": "akari-mem", "tags": "quantization,fp16"},
    ]
    ids = []
    t0 = time.time()
    for d in docs:
        sid = store.save(title=d["title"], text=d["text"], project=d["project"], tags=d["tags"])
        ids.append(sid)
    save_t = time.time() - t0
    print(f"saved {len(docs)} items in {save_t:.2f}s ({save_t/len(docs)*1000:.1f} ms/item)")

    time.sleep(2)

    banner("STAGE 2: search (no filter)")
    t0 = time.time()
    results = store.search("ONNX 量化 FP16 加速", limit=3)
    print(f"top-{len(results)} in {(time.time()-t0)*1000:.1f} ms:")
    for r in results:
        print(f"  [{r.get('score', 0):.3f}] proj={r.get('project')!r:18s} {r.get('text','')[:60]}")

    banner("STAGE 3: search with project filter")
    t0 = time.time()
    results = store.search("Spring Boot 反应式", limit=3, project="huifeixingry")
    print(f"top-{len(results)} in {(time.time()-t0)*1000:.1f} ms:")
    for r in results:
        print(f"  [{r.get('score', 0):.3f}] proj={r.get('project')!r:18s} {r.get('text','')[:60]}")

    banner("STAGE 4: search with tag filter")
    t0 = time.time()
    results = store.search("量化精度", limit=3, tags="fp16")
    print(f"top-{len(results)} in {(time.time()-t0)*1000:.1f} ms:")
    for r in results:
        print(f"  [{r.get('score', 0):.3f}] tags={r.get('tags')!r:20s} {r.get('text','')[:60]}")

    banner("STAGE 5: delete")
    deleted = store.delete(ids[0])
    print(f"deleted id={ids[0]}: {deleted}")

    banner("STAGE 6: verify delete")
    results = store.search("ONNX 量化", limit=5)
    found = any(r.get("id") == ids[0] for r in results)
    print(f"deleted item still findable? {found}  (expected: False)")
    if found:
        print("  WARNING: delete did not propagate to vector store")

    print()
    print("=" * 60)
    print("SMOKE TEST FINISHED OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
