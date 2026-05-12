"""Quality + VRAM verification for BGE-M3 ONNX FP16.

Compares dense embeddings against PyTorch BGE-M3 (sentence-transformers).
Reports:
  - cosine similarity per text (must be > 0.999 for production)
  - VRAM occupied by each backend
  - cold-start + warm-query latency
"""
import os
import sys
import time
import gc

os.environ.setdefault("HF_HOME", "F:/tmp/hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "F:/tmp/hf-cache")
os.environ.setdefault("AKARI_MODEL_CACHE", "F:/models")

import numpy as np

sys.path.insert(0, "F:/claude-tools/akari-mem-mcp")
from embeddings import OnnxEmbeddingProvider, LocalEmbeddingProvider


TEST_TEXTS = [
    "猫娘工程师浮浮酱在写 ONNX 转换脚本",
    "RAG retrieval-augmented generation with BGE-M3",
    "Spring Boot 3.x with reactive programming and WebFlux",
    "汇飞行管理系统的微信小程序登录功能",
    "Akari mem MCP — dual engine SQLite + ChromaDB store",
    "Today's weather is wonderful, perfect for a stroll outside.",
    "深度学习模型量化：从 FP32 到 FP16 的精度损失通常小于 0.1%",
    "短",
    "这是一段比较长的中文文本，目的是测试 BGE-M3 在处理较长输入时的稳定性，以及 FP16 量化后是否会引入数值上的偏移，进而影响下游 ChromaDB 的相似度排序。",
]


def vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except ImportError:
        pass
    return 0.0


def banner(s: str):
    print()
    print("=" * 64)
    print(s)
    print("=" * 64)


def main():
    banner("STAGE 1: ONNX FP16 (CUDA EP)")
    vram0 = vram_mb()
    onnx_p = OnnxEmbeddingProvider(
        model_path="F:/models/bge-m3-onnx-fp16",
        model_name="BAAI/bge-m3",
        device="cuda",
        max_length=512,
        normalize=True,
    )

    t0 = time.time()
    onnx_vecs = onnx_p.embed(TEST_TEXTS)
    cold_latency = time.time() - t0
    onnx_vecs_np = np.asarray(onnx_vecs, dtype=np.float32)
    print(f"cold-start + first embed: {cold_latency*1000:.1f} ms")
    print(f"output shape:            {onnx_vecs_np.shape}")
    print(f"VRAM after embed:        {vram_mb() - vram0:.1f} MB (delta)")

    t0 = time.time()
    for _ in range(5):
        onnx_p.embed(TEST_TEXTS)
    warm_latency = (time.time() - t0) / 5
    print(f"warm latency / batch:    {warm_latency*1000:.1f} ms ({warm_latency*1000/len(TEST_TEXTS):.2f} ms/text)")

    banner("STAGE 2: PyTorch BGE-M3 FP32 (baseline)")
    local_p = LocalEmbeddingProvider(
        model_name="BAAI/bge-m3",
        cache_dir="F:/models",
        device="cuda",
    )
    vram_pre_pt = vram_mb()
    t0 = time.time()
    local_vecs = local_p.embed(TEST_TEXTS)
    pt_first = time.time() - t0
    local_vecs_np = np.asarray(local_vecs, dtype=np.float32)
    print(f"first embed: {pt_first*1000:.1f} ms")
    print(f"VRAM after PT load+embed: {vram_mb():.1f} MB (delta from pre-PT={vram_mb()-vram_pre_pt:.1f})")

    banner("STAGE 3: Cosine similarity FP16 vs FP32")
    for i, txt in enumerate(TEST_TEXTS):
        cos = float(np.dot(onnx_vecs_np[i], local_vecs_np[i]))
        flag = "OK" if cos >= 0.999 else "WARN" if cos >= 0.99 else "BAD"
        snippet = txt if len(txt) <= 50 else txt[:47] + "..."
        print(f"  [{flag:4s}] cos={cos:.6f}   {snippet}")

    # Aggregate
    cos_all = np.array([
        float(np.dot(onnx_vecs_np[i], local_vecs_np[i]))
        for i in range(len(TEST_TEXTS))
    ])
    print()
    print(f"  min cos = {cos_all.min():.6f}")
    print(f"  avg cos = {cos_all.mean():.6f}")
    print(f"  max cos = {cos_all.max():.6f}")

    quality_ok = cos_all.min() >= 0.999
    print()
    print("VERDICT:", "PASS (FP16 quality is production-grade)" if quality_ok
          else "REVIEW (cos < 0.999 — investigate)")


if __name__ == "__main__":
    main()
