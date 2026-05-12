"""Convert BGE-M3 ONNX FP32 -> FP16 for VRAM savings.

Strategy:
- Load with external data (model.onnx_data is 2.1GB).
- Convert weights to float16 via onnxconverter_common.
- keep_io_types=True so input_ids (int64) and output (float32) stay unchanged,
  which means OnnxEmbeddingProvider does not need any I/O dtype handling.
- Save with new external data file to keep model.onnx small.

Expected VRAM reduction: ~2.1GB -> ~1.1GB.
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", "F:/tmp/hf-cache")

import onnx
# Use onnxruntime's Transformer-specific FP16 converter (handles BERT/RoBERTa
# style architectures robustly). onnxconverter_common.float16 had bugs with
# BGE-M3 (wiped all initializers on the converted ModelProto).
from onnxruntime.transformers.float16 import convert_float_to_float16

SRC_DIR = "F:/models/bge-m3-onnx-fp32"
DST_DIR = "F:/models/bge-m3-onnx-fp16"
SRC_MODEL = os.path.join(SRC_DIR, "onnx", "model.onnx")
DST_MODEL = os.path.join(DST_DIR, "onnx", "model.onnx")
DST_DATA = "model.onnx_data"

print("=" * 60)
print("BGE-M3 ONNX FP32 -> FP16 converter")
print("=" * 60)
print(f"Source: {SRC_MODEL}")
print(f"Target: {DST_MODEL}")
print()

if not os.path.exists(SRC_MODEL):
    print(f"ERROR: source model not found: {SRC_MODEL}", file=sys.stderr)
    sys.exit(1)

os.makedirs(os.path.dirname(DST_MODEL), exist_ok=True)

# BGE-M3 FP32 model is ~2.16GB (>2GB ProtoBuf hard limit). ort's path-based
# convert_float_to_float16 uses tempfile.NamedTemporaryFile which fails on
# Windows (handle is exclusively locked while infer_shapes_path tries to
# write). Workaround: run shape inference ourselves (close handle first),
# load as ModelProto with external_data, then call convert with
# disable_shape_infer=True.
import tempfile
from onnx.shape_inference import infer_shapes_path

src_dir = os.path.dirname(SRC_MODEL)

print("[1a]  Pre-running shape inference (manual, Windows-safe) ...")
fd, si_path = tempfile.mkstemp(prefix="bgem3_si_", suffix=".onnx", dir=src_dir)
os.close(fd)
try:
    t0 = time.time()
    infer_shapes_path(SRC_MODEL, si_path)
    print(f"      done in {time.time() - t0:.1f}s ({os.path.getsize(si_path)/1024:.1f} KB graph file)")

    print("[1b]  Loading shape-inferred ModelProto with external data ...")
    t0 = time.time()
    model_inferred = onnx.load(si_path, load_external_data=True)
    print(f"      done in {time.time() - t0:.1f}s "
          f"(#initializers = {len(model_inferred.graph.initializer)})")

    print("[1c]  Converting weights to FP16 (skip internal shape_infer) ...")
    t0 = time.time()
    model_fp16 = convert_float_to_float16(
        model_inferred,
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=None,
        node_block_list=None,
        force_fp16_initializers=True,
    )
    print(f"      done in {time.time() - t0:.1f}s")
    print(f"      #initializers after convert: {len(model_fp16.graph.initializer)}")
finally:
    if os.path.exists(si_path):
        try:
            os.unlink(si_path)
        except OSError:
            pass

print("[2/2] Saving FP16 model with external data...")
t0 = time.time()
onnx.save_model(
    model_fp16,
    DST_MODEL,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=DST_DATA,
    size_threshold=1024,
    convert_attribute=False,
)
print(f"      done in {time.time() - t0:.1f}s")

print("[3/3] Copying tokenizer & config files...")
import shutil
for name in (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
):
    src_a = os.path.join(SRC_DIR, "onnx", name)
    src_b = os.path.join(SRC_DIR, name)
    src = src_a if os.path.exists(src_a) else src_b if os.path.exists(src_b) else None
    if src:
        dst = os.path.join(DST_DIR, "onnx", name)
        shutil.copy2(src, dst)
        print(f"      {name:35s} OK")

print()
print("=== Size comparison ===")
fp32_data = os.path.getsize(os.path.join(SRC_DIR, "onnx", "model.onnx_data"))
fp16_data = os.path.getsize(os.path.join(DST_DIR, "onnx", DST_DATA))
print(f"FP32 weights: {fp32_data / 1024**2:.1f} MB")
print(f"FP16 weights: {fp16_data / 1024**2:.1f} MB ({100*fp16_data/fp32_data:.1f}% of FP32)")
print(f"Saved:        {(fp32_data - fp16_data) / 1024**2:.1f} MB")
print()
print(f"DONE -> {DST_DIR}")
