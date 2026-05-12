"""
Embedding providers for akari-mem-mcp.

Supports 5 modes:
1. LOCAL     — sentence-transformers (BGE-M3, etc.), best quality
2. ONNX      — onnxruntime-gpu (BGE-M3 FP16), best VRAM/speed balance
3. FASTEMBED — ONNX-based FastEmbed (BGE-small, etc.), lightweight
4. API       — OpenAI-compatible embedding API, zero local resources
5. DEFAULT   — ChromaDB built-in (all-MiniLM-L6-v2), fallback
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger("akari-mem.embeddings")


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding via sentence-transformers. Lazy-loads on first call."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "AKARI_MODEL_CACHE", "F:/models"
        )
        self._device = device  # None = auto-detect
        self._model = None
        self._dim: Optional[int] = None

    def _resolve_device(self) -> str:
        """Resolve device: explicit > env > auto-detect (CUDA > CPU)."""
        if self._device and self._device != "auto":
            return self._device
        env_device = os.environ.get("AKARI_DEVICE")
        if env_device:
            return env_device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _resolve_dtype(self, device: str):
        """Pick optimal dtype for the device: BF16 > FP16 > FP32."""
        if device != "cuda":
            return None  # CPU stays FP32
        try:
            import torch
            if torch.cuda.is_bf16_supported():
                logger.info("Using BF16 (Blackwell/Ampere native)")
                return torch.bfloat16
            else:
                logger.info("BF16 not supported, falling back to FP16")
                return torch.float16
        except Exception:
            return None

    def _load(self):
        if self._model is None:
            device = self._resolve_device()
            dtype = self._resolve_dtype(device)
            dtype_name = str(dtype).split(".")[-1] if dtype else "fp32"
            logger.info(f"Loading local model: {self._model_name} on device={device}, dtype={dtype_name} ...")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            model_kwargs = {}
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            self._model = SentenceTransformer(
                self._model_name, cache_folder=self._cache_dir, device=device,
                model_kwargs=model_kwargs,
            )
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded: dim={self._dim}, device={device}, dtype={dtype_name}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore

    @property
    def model_name(self) -> str:
        return self._model_name


class APIEmbeddingProvider(EmbeddingProvider):
    """
    Online embedding via OpenAI-compatible API.
    Works with: OpenAI, Azure OpenAI, Ollama, LiteLLM, vLLM, etc.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        import urllib.request

        payload = json.dumps({
            "model": self._model_name,
            "input": texts,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        sorted_data = sorted(result["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in sorted_data]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class DefaultEmbeddingProvider(EmbeddingProvider):
    """ChromaDB built-in (all-MiniLM-L6-v2). Zero-config fallback."""

    def __init__(self):
        self._fn = None
        self._dim = 384

    def _load(self):
        if self._fn is None:
            try:
                from chromadb.utils.embedding_functions import (
                    DefaultEmbeddingFunction,
                )
                self._fn = DefaultEmbeddingFunction()
            except ImportError:
                from chromadb.utils.embedding_functions import (
                    ONNXMiniLM_L6_V2,
                )
                self._fn = ONNXMiniLM_L6_V2()

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        # Use __call__ (batch API) — embed_query has broken behavior in v1.5.5
        raw = self._fn(texts)
        results = []
        for vec in raw:
            # Ensure pure Python list[float] for ChromaDB Rust bindings
            if hasattr(vec, 'tolist'):
                vec = vec.tolist()
            results.append(vec)
        return results

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "all-MiniLM-L6-v2"


class OnnxEmbeddingProvider(EmbeddingProvider):
    """
    ONNX-based embedding via onnxruntime (CUDA EP / CPU EP).

    Designed for BGE-M3 ONNX FP16 (self-converted) to minimize VRAM usage
    while keeping retrieval quality essentially unchanged vs PyTorch FP32.

    Expected layout under `model_path`:
        <model_path>/onnx/model.onnx              (graph)
        <model_path>/onnx/model.onnx_data         (external weights)
        <model_path>/onnx/tokenizer.json
        <model_path>/onnx/sentencepiece.bpe.model
        <model_path>/onnx/tokenizer_config.json
    """

    def __init__(
        self,
        model_path: str = "F:/models/bge-m3-onnx-fp16",
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        max_length: int = 512,
        normalize: bool = True,
        pooling: str = "cls",
    ):
        self._model_path = model_path
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._normalize = normalize
        self._pooling = pooling  # "cls" | "mean"
        self._session = None
        self._tokenizer = None
        self._dim: Optional[int] = None
        self._output_name: Optional[str] = None
        self._needs_token_type_ids: bool = False

    def _load(self):
        if self._session is not None:
            return
        import onnxruntime as ort

        onnx_dir = os.path.join(self._model_path, "onnx")
        onnx_file = os.path.join(onnx_dir, "model.onnx")
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(
                f"ONNX model file not found: {onnx_file}. "
                f"Did you run scripts/convert_bge_m3_fp16.py?"
            )

        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers required for ONNX BGE-M3 tokenizer."
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

        # Build provider list (CUDA first if requested, fallback to CPU).
        # IMPORTANT: arena_extend_strategy default is "kNextPowerOfTwo" which
        # over-allocates VRAM aggressively (e.g. doubles arena on each grow).
        # For inference workloads with fixed batch size this is wasteful.
        # `kSameAsRequested` keeps allocations tight; `HEURISTIC` for cuDNN
        # avoids the EXHAUSTIVE workspace bloat.
        providers = []
        if self._device == "cuda":
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            ))
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable mem-pattern (it pre-allocates worst-case across input shapes,
        # explodes VRAM for dynamic-length tokenizer output).
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False

        logger.info(
            f"Loading ONNX model: {onnx_file} "
            f"(requested providers={[p[0] if isinstance(p, tuple) else p for p in providers]})"
        )
        self._session = ort.InferenceSession(
            onnx_file, sess_options=sess_options, providers=providers
        )

        input_names = {i.name for i in self._session.get_inputs()}
        self._needs_token_type_ids = "token_type_ids" in input_names

        # Pick output: prefer dense embedding (1024) or pooler_output
        outputs = self._session.get_outputs()
        preferred_names = ("sentence_embedding", "pooler_output", "dense_vecs")
        for o in outputs:
            if o.name in preferred_names:
                self._output_name = o.name
                shape_last = o.shape[-1] if o.shape else None
                self._dim = shape_last if isinstance(shape_last, int) else 1024
                break
        if self._output_name is None:
            # Fallback: scan for a 2D output with last dim == 1024
            for o in outputs:
                if (
                    len(o.shape) == 2
                    and isinstance(o.shape[-1], int)
                    and o.shape[-1] == 1024
                ):
                    self._output_name = o.name
                    self._dim = 1024
                    break
        if self._output_name is None:
            # Last resort: take last_hidden_state and we'll pool ourselves
            self._output_name = outputs[0].name
            self._dim = 1024

        actual = self._session.get_providers()
        logger.info(
            f"ONNX session ready: dim={self._dim}, output='{self._output_name}', "
            f"providers={actual}, token_type_ids={self._needs_token_type_ids}"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np

        self._load()

        enc = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        feed = {
            "input_ids": enc["input_ids"].astype("int64"),
            "attention_mask": enc["attention_mask"].astype("int64"),
        }
        if self._needs_token_type_ids:
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype="int64")

        outputs = self._session.run([self._output_name], feed)
        emb = outputs[0]

        # If we got last_hidden_state (shape [B, T, H]), apply pooling
        if emb.ndim == 3:
            if self._pooling == "mean":
                mask = feed["attention_mask"][..., None].astype("float32")
                emb = (emb * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            else:  # cls
                emb = emb[:, 0, :]

        emb = emb.astype("float32")
        if self._normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms

        return emb.tolist()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore

    @property
    def model_name(self) -> str:
        return self._model_name


class FastEmbedProvider(EmbeddingProvider):
    """
    Lightweight ONNX-based embedding via FastEmbed (by Qdrant).
    Uses quantized ONNX models — much lower memory and faster CPU inference
    compared to sentence-transformers (PyTorch). Lazy-loads on first call.

    Supported models include BAAI/bge-small-zh-v1.5, jinaai/jina-embeddings-v2-base-zh, etc.
    """

    # Known model dimensions (avoids loading model just to check dim)
    _KNOWN_DIMS = {
        "BAAI/bge-small-zh-v1.5": 512,
        "jinaai/jina-embeddings-v2-base-zh": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get(
            "AKARI_MODEL_CACHE", "F:/models/fastembed"
        )
        self._model = None
        self._dim = self._KNOWN_DIMS.get(model_name)

    def _load(self):
        if self._model is None:
            logger.info(f"Loading FastEmbed model: {self._model_name} ...")
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise RuntimeError(
                    "fastembed not installed. "
                    "Run: pip install fastembed"
                )
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
            )
            # Probe dimension if not known
            if self._dim is None:
                probe = list(self._model.embed(["dim probe"]))
                self._dim = len(probe[0])
            logger.info(
                f"FastEmbed model loaded: {self._model_name} (dim={self._dim})"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        # FastEmbed.embed() returns a generator of numpy arrays
        return [vec.tolist() for vec in self._model.embed(texts)]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore

    @property
    def model_name(self) -> str:
        return self._model_name


class ChromaEmbeddingAdapter:
    """
    Adapter that wraps EmbeddingProvider into ChromaDB's EmbeddingFunction
    protocol. Supports both v0.x (__call__) and v1.x (embed_query/embed_documents).
    """

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def __call__(self, input: List[str]) -> List[List[float]]:
        vecs = self.provider.embed(input)
        # Ensure pure Python list for ChromaDB Rust bindings
        return [v.tolist() if hasattr(v, 'tolist') else v for v in vecs]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """ChromaDB v1.x: embed documents for storage."""
        vecs = self.provider.embed(documents)
        return [v.tolist() if hasattr(v, 'tolist') else v for v in vecs]

    def embed_query(self, input: str) -> List[float]:
        """ChromaDB v1.x: embed a single query for search."""
        vec = self.provider.embed([input])[0]
        return vec.tolist() if hasattr(vec, 'tolist') else vec

    def name(self) -> str:
        """Required by ChromaDB to identify the embedding function."""
        return f"akari_mem_{self.provider.model_name}"


def create_provider(config: dict) -> EmbeddingProvider:
    """
    Factory: create embedding provider from config dict.

    Config examples:
      {"mode": "local", "model": "BAAI/bge-m3"}
      {"mode": "onnx", "model_path": "F:/models/bge-m3-onnx-fp16",
       "model": "BAAI/bge-m3", "device": "cuda"}
      {"mode": "fastembed", "model": "BAAI/bge-small-zh-v1.5"}
      {"mode": "api", "url": "https://api.openai.com/v1/embeddings",
       "key": "sk-...", "model": "text-embedding-3-small", "dim": 1536}
      {"mode": "default"}
    """
    mode = config.get("mode", "default")

    if mode == "local":
        return LocalEmbeddingProvider(
            model_name=config.get("model", "BAAI/bge-m3"),
            cache_dir=config.get("cache_dir"),
            device=config.get("device"),
        )
    elif mode == "onnx":
        return OnnxEmbeddingProvider(
            model_path=config.get("model_path", "F:/models/bge-m3-onnx-fp16"),
            model_name=config.get("model", "BAAI/bge-m3"),
            device=config.get("device", "cuda"),
            max_length=int(config.get("max_length", 512)),
            normalize=bool(config.get("normalize", True)),
            pooling=config.get("pooling", "cls"),
        )
    elif mode == "fastembed":
        return FastEmbedProvider(
            model_name=config.get("model", "BAAI/bge-small-zh-v1.5"),
            cache_dir=config.get("cache_dir"),
        )
    elif mode == "api":
        return APIEmbeddingProvider(
            api_url=config["url"],
            api_key=config.get("key", ""),
            model_name=config.get("model", "text-embedding-3-small"),
            dimension=config.get("dim", 1536),
        )
    else:
        return DefaultEmbeddingProvider()
