# -*- coding: utf-8 -*-
"""
Test: simulate exactly what save_memory does, with timing at each stage.
"""
import sys, os, time
_PROJECT_ROOT = r"F:\claude-tools\akari-mem-mcp"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env_loader import setup
setup()

import asyncio

async def main():
    print("=== save_memory Simulation ===\n")
    
    # Simulate what get_store() + save() does
    def full_save_simulation():
        t0 = time.time()
        
        print(f"  [1] Creating embedding provider...")
        t1 = time.time()
        from embeddings import create_provider
        config = {"mode": "local", "model": "BAAI/bge-m3"}
        provider = create_provider(config)
        print(f"      Provider created: {time.time()-t1:.2f}s")
        
        print(f"  [2] Creating reranker...")
        t1 = time.time()
        from rerank import create_reranker
        reranker = create_reranker({"mode": "local", "model": "BAAI/bge-reranker-v2-m3"})
        print(f"      Reranker created: {time.time()-t1:.2f}s")
        
        print(f"  [3] Creating MemoryStore (SQLite + ChromaDB init)...")
        t1 = time.time()
        from store import MemoryStore
        store = MemoryStore(
            data_dir=os.path.join(_PROJECT_ROOT, "data"),
            embedding_provider=provider,
            reranker=reranker
        )
        print(f"      MemoryStore ready: {time.time()-t1:.2f}s")
        
        print(f"  [4] Saving test memory...")
        t1 = time.time()
        mem_id = store.save("async test", "testing save from simulation", tags="test", source="test")
        print(f"      Saved #{mem_id}: {time.time()-t1:.2f}s")
        
        # cleanup
        store.delete(mem_id)
        print(f"      Cleaned up #{mem_id}")
        
        total = time.time() - t0
        print(f"\n  Total time: {total:.2f}s")
        return mem_id
    
    print("Running in asyncio.to_thread...")
    t0 = time.time()
    result = await asyncio.to_thread(full_save_simulation)
    print(f"\nto_thread completed in {time.time()-t0:.2f}s")

asyncio.run(main())
print("\n=== Done ===")
