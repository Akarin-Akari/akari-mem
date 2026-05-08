import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embeddings import create_provider
from rerank import create_reranker

def test_fastembed_embedding():
    print("--- Testing FastEmbed Provider ---")
    config = {
        "mode": "fastembed",
        "model": "BAAI/bge-small-zh-v1.5"
    }
    provider = create_provider(config)
    print(f"Model Name: {provider.model_name}")
    print(f"Expected Dimension: {provider.dimension}")
    
    texts = ["这是一个测试。", "FastEmbed is fast!"]
    embeddings = provider.embed(texts)
    
    print(f"Embedded {len(embeddings)} texts.")
    print(f"Dimension of first embedding: {len(embeddings[0])}")
    print(f"First few values of first embedding: {embeddings[0][:5]}")
    
    assert len(embeddings) == 2, "Expected 2 embeddings"
    assert len(embeddings[0]) == provider.dimension, "Dimension mismatch"
    assert isinstance(embeddings[0][0], float), "Expected float values"
    print("[SUCCESS] FastEmbed embedding tests passed!")

def test_fastembed_reranker():
    print("\n--- Testing FastEmbed Reranker ---")
    config = {
        "mode": "fastembed",
        "model": "jinaai/jina-reranker-v2-base-multilingual"
    }
    reranker = create_reranker(config)
    print(f"Model Name: {reranker.model_name}")
    
    query = "如何使用FastEmbed？"
    documents = [
        {"title": "Sentence Transformers", "text": "It uses PyTorch and is very flexible."},
        {"title": "FastEmbed Guide", "text": "FastEmbed is a lightweight, fast, Python library built for embedding generation. 它可以很快生成向量。"},
        {"title": "Cooking Recipe", "text": "Add salt and pepper to taste."}
    ]
    
    reranked = reranker.rerank(query, documents, top_k=2)
    
    print(f"Reranked {len(reranked)} documents out of {len(documents)}.")
    for i, doc in enumerate(reranked):
        print(f"Rank {i+1}: Score {doc['rerank_score']:.4f} | {doc['title']}")
        
    assert len(reranked) == 2, "Expected 2 reranked documents"
    assert "rerank_score" in reranked[0], "Missing rerank_score"
    assert reranked[0]["title"] == "FastEmbed Guide", "Expected FastEmbed Guide to be top ranked"
    print("[SUCCESS] FastEmbed reranker tests passed!")

if __name__ == "__main__":
    try:
        test_fastembed_embedding()
        test_fastembed_reranker()
        print("\n[DONE] All FastEmbed Phase 3 tests passed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {repr(e)}")
        sys.exit(1)
