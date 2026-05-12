"""Measure REAL VRAM of PyTorch BGE-M3 (current production setup)."""
import os
import sys
import time
import subprocess

os.environ.setdefault("HF_HOME", "F:/tmp/hf-cache")
os.environ.setdefault("AKARI_MODEL_CACHE", "F:/models")
sys.path.insert(0, "F:/claude-tools/akari-mem-mcp")


def nvidia_smi_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        encoding="utf-8",
    )
    return float(out.strip().splitlines()[0])


def main():
    print(f"[t=0]    baseline VRAM: {nvidia_smi_mb():.0f} MB")

    from embeddings import LocalEmbeddingProvider
    print(f"[t=after imports]      VRAM: {nvidia_smi_mb():.0f} MB")

    p = LocalEmbeddingProvider(
        model_name="BAAI/bge-m3",
        cache_dir="F:/models",
        device="cuda",
    )
    _ = p.embed(["warm-up text 1", "warm-up text 2"])
    time.sleep(1)
    print(f"[t=after PT load+1]   VRAM: {nvidia_smi_mb():.0f} MB")

    texts = ["sample text " + str(i) for i in range(32)]
    _ = p.embed(texts)
    time.sleep(1)
    print(f"[t=after batch 32]    VRAM: {nvidia_smi_mb():.0f} MB")


if __name__ == "__main__":
    main()
