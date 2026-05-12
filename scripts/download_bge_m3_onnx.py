"""Download BAAI/bge-m3 ONNX subdirectory to F:/models/bge-m3-onnx-fp32"""
import os
import sys

os.environ["HF_HOME"] = "F:/tmp/hf-cache"
os.environ["HF_HUB_CACHE"] = "F:/tmp/hf-cache"

from huggingface_hub import snapshot_download

print("=" * 60)
print("Downloading BAAI/bge-m3 ONNX files to F:/models/bge-m3-onnx-fp32")
print("=" * 60)
print()

target_dir = "F:/models/bge-m3-onnx-fp32"
os.makedirs(target_dir, exist_ok=True)

try:
    local_path = snapshot_download(
        repo_id="BAAI/bge-m3",
        allow_patterns=[
            "onnx/*",
            "tokenizer.json",
            "tokenizer_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "config.json",
        ],
        local_dir=target_dir,
        cache_dir="F:/tmp/hf-cache",
    )
    print(f"\nSUCCESS! Downloaded to: {local_path}")

    print("\n=== Downloaded files ===")
    total_size = 0
    for root, _, files in os.walk(target_dir):
        for f in files:
            full = os.path.join(root, f)
            size_mb = os.path.getsize(full) / 1024**2
            total_size += size_mb
            print(f"  {os.path.relpath(full, target_dir):50s} | {size_mb:>8.2f} MB")
    print(f"\nTotal: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
